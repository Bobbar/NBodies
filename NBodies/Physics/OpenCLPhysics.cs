using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Threading;
using Cloo;
using Cloo.Bindings;
using Cloo.Extensions;
using System.IO;
using NBodies.Extensions;
using System.Runtime.InteropServices;
using System.Numerics;

namespace NBodies.Physics
{
    public class OpenCLPhysics : IPhysicsCalc, IDisposable
    {
        private int _gpuIndex = 4;
        private int _levels = 4;
        private static int _threadsPerBlock = 256;
        private int _parallelPartitions = 12;//24;
        private long _maxBufferSize = 0;

        private int[] _levelIdx = new int[0];
        private MeshCell[] _mesh = new MeshCell[0];
        private int _meshLength = 0;
        private GridInfo[] _gridInfo = new GridInfo[0];
        private Body[] _bodies = new Body[0];
        private static Body[] _sortBodies = new Body[0];
        private static SpatialInfo[] _spatials = new SpatialInfo[0];
        private static int[] _mortKeys = new int[0];
        private static int[] _cellIdx = new int[0];

        private ComputeContext _context;
        private ComputeCommandQueue _queue;

        private ComputeProgram _program;

        private ComputeKernel _forceKernel;
        private ComputeKernel _collisionKernel;
        private ComputeKernel _collisionLargeKernel;
        private ComputeKernel _popGridKernel;
        private ComputeKernel _clearGridKernel;
        private ComputeKernel _buildNeighborsKernel;
        private ComputeKernel _fixOverlapKernel;
        private ComputeKernel _buildBottomKernel;
        private ComputeKernel _buildTopKernel;
        private ComputeKernel _calcCMKernel;

        private ComputeBuffer<int> _gpuLevelIdx;
        private ComputeBuffer<MeshCell> _gpuMesh;
        private ComputeBuffer<int> _gpuMeshNeighbors;
        private ComputeBuffer<Body> _gpuInBodies;
        private ComputeBuffer<Body> _gpuOutBodies;
        private ComputeBuffer<int> _gpuGridIndex;
        private ComputeBuffer<Vector2> _gpuCM;

        private static Dictionary<long, BufferDims> _bufferInfo = new Dictionary<long, BufferDims>();

        private Stopwatch timer = new Stopwatch();
        private Stopwatch timer2 = new Stopwatch();


        public MeshCell[] CurrentMesh
        {
            get
            {
                return _mesh;
            }
        }

        public int[] LevelIndex
        {
            get
            {
                return _levelIdx;
            }
        }

        public OpenCLPhysics(int gpuIdx, int threadsperblock)
        {
            if (gpuIdx != -1)
                _gpuIndex = gpuIdx;

            if (threadsperblock != -1)
                _threadsPerBlock = threadsperblock;
        }

        public void Init()
        {
            var devices = GetDevices();

            var device = devices[_gpuIndex];
            var platform = device.Platform;

            _maxBufferSize = device.MaxMemoryAllocationSize;
            _context = new ComputeContext(new[] { device }, new ComputeContextPropertyList(platform), null, IntPtr.Zero);
            _queue = new ComputeCommandQueue(_context, device, ComputeCommandQueueFlags.None);

            StreamReader streamReader = new StreamReader(Environment.CurrentDirectory + "/Physics/Kernels.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            _program = new ComputeProgram(_context, clSource);

            try
            {
                _program.Build(null, "-cl-std=CL1.2", null, IntPtr.Zero);
            }
            catch (BuildProgramFailureComputeException ex)
            {
                string buildLog = _program.GetBuildLog(device);
                System.IO.File.WriteAllText("build_error.txt", buildLog);
                Console.WriteLine(buildLog);
                throw;
            }


            Console.WriteLine(_program.GetBuildLog(device));

            _forceKernel = _program.CreateKernel("CalcForce");
            _collisionKernel = _program.CreateKernel("CalcCollisions");
            _collisionLargeKernel = _program.CreateKernel("CalcCollisionsLarge");
            _popGridKernel = _program.CreateKernel("PopGrid");
            _buildNeighborsKernel = _program.CreateKernel("BuildNeighbors");
            _clearGridKernel = _program.CreateKernel("ClearGrid");
            _fixOverlapKernel = _program.CreateKernel("FixOverlaps");

            _buildBottomKernel = _program.CreateKernel("BuildBottom");
            _buildTopKernel = _program.CreateKernel("BuildTop");
            _calcCMKernel = _program.CreateKernel("CalcCenterOfMass");

            InitBuffers();
        }

        private List<ComputeDevice> GetDevices()
        {
            var devices = new List<ComputeDevice>();

            foreach (var platform in ComputePlatform.Platforms)
            {
                foreach (var device in platform.Devices)
                {
                    devices.Add(device);
                }
            }

            return devices;
        }

        public void Flush()
        {
            _gpuLevelIdx.Dispose();
            _gpuMesh.Dispose();
            _gpuMeshNeighbors.Dispose();
            _gpuInBodies.Dispose();
            _gpuOutBodies.Dispose();
            _gpuGridIndex.Dispose();
            _gpuCM.Dispose();

            _bufferInfo.Clear();

            InitBuffers();
        }

        private void InitBuffers()
        {
            _gpuGridIndex = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1, IntPtr.Zero);
            Allocate(ref _gpuGridIndex, 0);

            _gpuMeshNeighbors = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1, IntPtr.Zero);
            Allocate(ref _gpuMeshNeighbors, 0);

            _gpuInBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, 1, IntPtr.Zero);
            Allocate(ref _gpuInBodies, 0, true);

            _gpuOutBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, 1, IntPtr.Zero);
            Allocate(ref _gpuOutBodies, 0, true);

            _gpuMesh = new ComputeBuffer<MeshCell>(_context, ComputeMemoryFlags.ReadWrite, 1, IntPtr.Zero);
            Allocate(ref _gpuMesh, 0, true);

            _gpuLevelIdx = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadOnly, 1, IntPtr.Zero);
            Allocate(ref _gpuLevelIdx, 0, true);


            _gpuCM = new ComputeBuffer<Vector2>(_context, ComputeMemoryFlags.ReadWrite, 1, IntPtr.Zero);
        }


        public void CalcMovement(ref Body[] bodies, float timestep, float viscosity, int cellSizeExp, float cullDistance, bool collisions, int meshLevels, int threadsPerBlock)
        {
            _bodies = bodies;
            _threadsPerBlock = threadsPerBlock;
            _levels = meshLevels;
            int threadBlocks = 0;

            // Calc number of thread blocks to fit the dataset.
            threadBlocks = BlockCount(_bodies.Length);

            // Build the particle mesh, mesh index, and mesh neighbors index.
            BuildMesh(cellSizeExp);

            // Calc center of mass on GPU.
            _calcCMKernel.SetMemoryArgument(0, _gpuMesh);
            _calcCMKernel.SetMemoryArgument(1, _gpuCM);
            _calcCMKernel.SetValueArgument(2, _levelIdx[_levels]);
            _calcCMKernel.SetValueArgument(3, _meshLength);

            _queue.ExecuteTask(_calcCMKernel, null);

            // Allocate and write the level index.
            Allocate(ref _gpuLevelIdx, _levelIdx.Length, true);
            _queue.WriteToBuffer(_levelIdx, _gpuLevelIdx, false, null);

            int argi = 0;
            _forceKernel.SetMemoryArgument(argi++, _gpuOutBodies);
            _forceKernel.SetValueArgument(argi++, _bodies.Length);
            _forceKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _forceKernel.SetMemoryArgument(argi++, _gpuMesh);
            _forceKernel.SetValueArgument(argi++, _meshLength);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _forceKernel.SetValueArgument(argi++, timestep);
            _forceKernel.SetValueArgument(argi++, _levels);
            _forceKernel.SetMemoryArgument(argi++, _gpuLevelIdx);

            _queue.Execute(_forceKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);

            argi = 0;
            _collisionLargeKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _collisionLargeKernel.SetValueArgument(argi++, _bodies.Length);
            _collisionLargeKernel.SetMemoryArgument(argi++, _gpuOutBodies);
            _collisionLargeKernel.SetMemoryArgument(argi++, _gpuMesh);
            _collisionLargeKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _collisionLargeKernel.SetValueArgument(argi++, Convert.ToInt32(collisions));

            _queue.Execute(_collisionLargeKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);

            argi = 0;
            _collisionKernel.SetMemoryArgument(argi++, _gpuOutBodies);
            _collisionKernel.SetValueArgument(argi++, _bodies.Length);
            _collisionKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _collisionKernel.SetMemoryArgument(argi++, _gpuMesh);
            _collisionKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _collisionKernel.SetValueArgument(argi++, timestep);
            _collisionKernel.SetValueArgument(argi++, viscosity);
            _collisionKernel.SetMemoryArgument(argi++, _gpuCM);
            _collisionKernel.SetValueArgument(argi++, cullDistance);
            _collisionKernel.SetValueArgument(argi++, Convert.ToInt32(collisions));

            _queue.Execute(_collisionKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);

            _queue.ReadFromBuffer(_gpuInBodies, ref bodies, true, null);
            _queue.Finish();

            if (_mesh.Length != _meshLength)
                _mesh = new MeshCell[_meshLength];

            _queue.ReadFromBuffer(_gpuMesh, ref _mesh, false, null);

        }


        public void FixOverLaps(ref Body[] bodies)
        {
            using (var inBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, bodies.Length, IntPtr.Zero))
            using (var outBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, bodies.Length, IntPtr.Zero))
            {
                _queue.WriteToBuffer(bodies, inBodies, true, null);
                _queue.Finish();

                _fixOverlapKernel.SetMemoryArgument(0, inBodies);
                _fixOverlapKernel.SetValueArgument(1, bodies.Length);
                _fixOverlapKernel.SetMemoryArgument(2, outBodies);

                _queue.Execute(_fixOverlapKernel, null, new long[] { BlockCount(bodies.Length) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                _queue.Finish();

                bodies = ReadBuffer(outBodies);
            }
        }

        /// <summary>
        /// Calculates number of thread blocks needed to fit the specified data length and the specified number of threads per block.
        /// </summary>
        /// <param name="len">Length of data set.</param>
        /// <param name="threads">Number of threads per block.</param>
        /// <returns>Number of blocks.</returns>
        public static int BlockCount(int len, int threads = 0)
        {
            if (threads == 0)
                threads = _threadsPerBlock;

            var blocks = (int)Math.Round((len - 1 + threads - 1) / (float)threads, 0);

            if (((threads * blocks) - len) > threads)
            {
                blocks -= 1;
            }
            else if ((threads * blocks) < len)
            {
                blocks += 1;
            }

            return blocks;
        }


        /// <summary>
        /// Calculate dimensionless morton number from X/Y coords.
        /// </summary>
        private int MortonNumber(int x, int y)
        {
            x &= 65535;
            x = (x | (x << 8)) & 16711935;
            x = (x | (x << 4)) & 252645135;
            x = (x | (x << 2)) & 858993459;
            x = (x | (x << 1)) & 1431655765;

            y &= 65535;
            y = (y | (y << 8)) & 16711935;
            y = (y | (y << 4)) & 252645135;
            y = (y | (y << 2)) & 858993459;
            y = (y | (y << 1)) & 1431655765;

            return x | (y << 1);
        }

        private void AddGridDims(MinMax minMax, int level)
        {
            long offsetX = (minMax.MinX - 1) * -1;
            long offsetY = (minMax.MinY - 1) * -1;

            long columns = Math.Abs(minMax.MinX - minMax.MaxX - 1);
            long rows = Math.Abs(minMax.MinY - minMax.MaxY - 1);

            long idxOff = 0;

            if (minMax.MinX < -400000 || minMax.MinY < -400000)
            {
                Debugger.Break();
            }

            if (level == 1)
            {
                idxOff = _gridInfo[level - 1].Size;
            }
            else if (level > 1)
            {
                idxOff = _gridInfo[level - 1].IndexOffset + _gridInfo[level - 1].Size;
            }

            _gridInfo[level] = new GridInfo(offsetX, offsetY, idxOff, minMax.MinX, minMax.MinY, minMax.MaxX, minMax.MaxY, columns, rows);

        }

        /// <summary>
        /// Builds the particle mesh and mesh-neighbor index for the current field.  Also begins writing the body array to GPU...
        /// </summary>
        /// <param name="cellSizeExp">Cell size exponent. 2 ^ exponent = cell size.</param>
        private void BuildMesh(int cellSizeExp)
        {
            // Grid info for each level.
            _gridInfo = new GridInfo[_levels + 1];

            // Get spatial info for the cells about to be constructed.
            LevelInfo botLevel = CalcBodySpatials(cellSizeExp);

            // At this point we are done modifying the body array,
            // so go ahead and start writing it to the GPU with a non-blocking call.
            Allocate(ref _gpuInBodies, _bodies.Length, true);
            Allocate(ref _gpuOutBodies, _bodies.Length, true);
            _queue.WriteToBuffer(_bodies, _gpuInBodies, false, null);

            LevelInfo[] topLevels = CalcTopSpatials(botLevel);

            // Get the total number of mesh cells to be created.
            int totCells = 0;
            foreach (var lvl in topLevels)
            {
                totCells += lvl.CellCount;
            }

            _meshLength = totCells;

            //// Reallocate the local mesh array as needed.
            //if (_mesh.Length != totCells)
            //    _mesh = new MeshCell[totCells];

            Allocate(ref _gpuMesh, totCells, true);

            // Index to hold the starting indexes for each level within the 1D mesh array.
            _levelIdx = new int[_levels + 1];
            _levelIdx[0] = 0;

            // Build the first (bottom) level of the mesh.
            BuildBottomLevelGPU(botLevel, cellSizeExp);

            // Build the remaining (top) levels of the mesh.
            BuildTopLevelsGPU(topLevels, cellSizeExp, _levels);

            // Populate the grid index and mesh neighbor index.
            PopGridAndNeighborsGPU(_gridInfo, totCells);
        }

        /// <summary>
        /// Computes spatial info (Morton number, X/Y indexes, mesh cell count) for all bodies.
        /// </summary>
        /// <param name="cellSizeExp">Cell size exponent. "Math.Pow(2, exponent)"</param>
        private LevelInfo CalcBodySpatials(int cellSizeExp)
        {
            // Spatial info to be computed.
            if (_spatials.Length != _bodies.Length)
                _spatials = new SpatialInfo[_bodies.Length];

            // Array of morton numbers used for sorting.
            // Using a key array when sorting is much faster than sorting an array of objects by a field.
            if (_mortKeys.Length != _bodies.Length)
                _mortKeys = new int[_bodies.Length];

            var minMax = new MinMax(0);
            var sync = new object();

            int pLen, pRem, pCount;
            Partition(_bodies.Length, _parallelPartitions, out pLen, out pRem, out pCount);

            // Compute the spatial info in parallel.
            Parallel.For(0, pCount, (p) =>
            {
                var mm = new MinMax(0);

                int offset = p * pLen;
                int len = offset + pLen;

                if (p == pCount - 1)
                    len += pRem;

                for (int b = offset; b < len; b++)
                {
                    int idxX = (int)_bodies[b].PosX >> cellSizeExp;
                    int idxY = (int)_bodies[b].PosY >> cellSizeExp;
                    int morton = MortonNumber(idxX, idxY);

                    mm.Update(idxX, idxY);

                    _spatials[b] = new SpatialInfo(morton, idxX, idxY, b);
                    _mortKeys[b] = morton;
                }

                lock (sync)
                {
                    minMax.Update(mm);
                }

            });

            AddGridDims(minMax, 0);

            // Sort by morton number to produce a spatially sorted array.
            Array.Sort(_mortKeys, _spatials);

            // Build a new sorted body array from the sorted spatial info.
            if (_sortBodies.Length != _bodies.Length)
                _sortBodies = new Body[_bodies.Length];

            Partition(_spatials.Length, _parallelPartitions, out pLen, out pRem, out pCount);
            Parallel.For(0, pCount, (p) =>
            {
                int offset = p * pLen;
                int len = offset + pLen;

                if (p == pCount - 1)
                    len += pRem;

                for (int b = offset; b < len; b++)
                {
                    _sortBodies[b] = _bodies[_spatials[b].Index];
                }

            });

            // Update the original body array with the sorted one.
            _bodies = _sortBodies;

            // Compute number of unique morton numbers to determine cell count,
            // and build the start index of each cell.
            int count = 0;
            int val = 0;

            //if (_cellIdx.Length != _bodies.Length + 1)
            //    _cellIdx = new int[_bodies.Length + 1];

            if (_cellIdx.Length < _bodies.Length)
                _cellIdx = new int[_bodies.Length + 100];

            var idx = new List<Vector2>(_bodies.Length);

            for (int i = 0; i < _spatials.Length; i++)
            {
                var spat = _spatials[i];
                // Update spatials index to match new sorted bodies.
                _spatials[i].Index = i;

                // Find the start of each new morton number and record location to build cell index.
                if (val != spat.Mort)
                {
                    _cellIdx[count] = i;
                    val = spat.Mort;

                    idx.Add(new Vector2(spat.IdxX, spat.IdxY));

                    count++;
                }
            }

            _cellIdx[count] = _spatials.Length;

            var output = new LevelInfo();
            output.Spatials = _spatials;
            output.CellCount = count;
            output.CellIndex = new int[count + 1];
            output.LocIdx = idx.ToArray();
            Array.Copy(_cellIdx, 0, output.CellIndex, 0, count + 1);

            return output;
        }

        /// <summary>
        /// Computes spatial info (Morton number, X/Y indexes, mesh cell count) for all top mesh levels.
        /// </summary>
        /// <param name="bottom">LevelInfo for bottom-most mesh level.</param>
        private LevelInfo[] CalcTopSpatials(LevelInfo bottom)
        {
            LevelInfo[] output = new LevelInfo[_levels + 1];

            output[0] = bottom;

            for (int level = 1; level <= _levels; level++)
            {
                object sync = new object();
                MinMax minMax = new MinMax(0);

                LevelInfo current = output[level - 1];

                output[level] = new LevelInfo();
                output[level].Spatials = new SpatialInfo[current.CellCount];

                int pLen, pRem, pCount;
                Partition(current.CellCount, _parallelPartitions, out pLen, out pRem, out pCount);

                Parallel.For(0, pCount, (p) =>
                {
                    var mm = new MinMax(0);

                    int offset = p * pLen;
                    int len = offset + pLen;

                    if (p == pCount - 1)
                        len += pRem;

                    for (int b = offset; b < len; b++)
                    {
                        var spatial = current.Spatials[current.CellIndex[b]];
                        int idxX = spatial.IdxX >> 1;
                        int idxY = spatial.IdxY >> 1;
                        int morton = MortonNumber(idxX, idxY);

                        mm.Update(idxX, idxY);

                        output[level].Spatials[b] = new SpatialInfo(morton, idxX, idxY, spatial.Index + b);
                    }

                    lock (sync)
                    {
                        minMax.Update(mm);
                    }

                });

                AddGridDims(minMax, level);

                int count = 0;
                int val = int.MaxValue;

                var idx = new List<Vector2>(current.CellCount);

                for (int i = 0; i < output[level].Spatials.Length; i++)
                {
                    var spat = output[level].Spatials[i];

                    if (val != spat.Mort)
                    {
                        _cellIdx[count] = i;
                        val = spat.Mort;

                        idx.Add(new Vector2(spat.IdxX, spat.IdxY));

                        count++;
                    }
                }

                _cellIdx[count] = output[level].Spatials.Length;

                output[level].CellCount = count;
                output[level].CellIndex = new int[count + 1];
                output[level].LocIdx = idx.ToArray();
                Array.Copy(_cellIdx, 0, output[level].CellIndex, 0, count + 1);
            }

            return output;
        }

        private void BuildBottomLevelGPU(LevelInfo levelInfo, int cellSizeExp)
        {
            int cellCount = levelInfo.CellCount;
            int cellSize = (int)Math.Pow(2, cellSizeExp);

            using (var gpuCellIdx = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadOnly, levelInfo.CellIndex.Length, IntPtr.Zero))
            using (var gpuLocIdx = new ComputeBuffer<Vector2>(_context, ComputeMemoryFlags.ReadOnly, levelInfo.LocIdx.Length, IntPtr.Zero))
            {
                _queue.WriteToBuffer(levelInfo.CellIndex, gpuCellIdx, false, null);
                _queue.WriteToBuffer(levelInfo.LocIdx, gpuLocIdx, false, null);

                _buildBottomKernel.SetMemoryArgument(0, _gpuInBodies);
                _buildBottomKernel.SetMemoryArgument(1, _gpuOutBodies);
                _buildBottomKernel.SetMemoryArgument(2, _gpuMesh);
                _buildBottomKernel.SetValueArgument(3, cellCount);
                _buildBottomKernel.SetMemoryArgument(4, gpuCellIdx);
                _buildBottomKernel.SetMemoryArgument(5, gpuLocIdx);
                _buildBottomKernel.SetValueArgument(6, cellSize);

                _queue.Execute(_buildBottomKernel, null, new long[] { BlockCount(cellCount) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
            }
        }

        private void BuildTopLevelsGPU(LevelInfo[] levelInfo, int cellSizeExp, int levels)
        {
            // Writing the cell and location indexes as single large arrays
            // is much faster than chunking them in at each level.

            // Calc total size of cell and location indexes.
            long cellIdxLen = 0;
            long locIdxLen = 0;

            for (int i = 1; i < levelInfo.Length; i++)
            {
                var lvl = levelInfo[i];

                cellIdxLen += lvl.CellIndex.Length;
                locIdxLen += lvl.LocIdx.Length;
            }

            // Build 1D arrays of cell and location indexes.
            var cellIdx = new int[cellIdxLen];
            var locIdx = new Vector2[locIdxLen];

            long cellIdxPos = 0;
            long locIdxPos = 0;

            for (int i = 1; i < levelInfo.Length; i++)
            {
                var lvl = levelInfo[i];

                Array.Copy(lvl.CellIndex, 0, cellIdx, cellIdxPos, lvl.CellIndex.Length);
                cellIdxPos += lvl.CellIndex.Length;

                Array.Copy(lvl.LocIdx, 0, locIdx, locIdxPos, lvl.LocIdx.Length);
                locIdxPos += lvl.LocIdx.Length;
            }

            // Allocate and write to the cell and location buffers.
            using (var gpuCellIdx = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadOnly, cellIdxLen, IntPtr.Zero))
            using (var gpuLocIdx = new ComputeBuffer<Vector2>(_context, ComputeMemoryFlags.ReadOnly, locIdxLen, IntPtr.Zero))
            {
                _queue.WriteToBuffer(cellIdx, gpuCellIdx, false, null);
                _queue.WriteToBuffer(locIdx, gpuLocIdx, false, null);

                int meshOffset = 0; // Write offset for new cells array location.
                int readOffset = 0; // Read offset for cell and location indexes.

                for (int level = 1; level <= levels; level++)
                {
                    int cellSizeExpLevel = cellSizeExp + level;
                    int cellSize = (int)Math.Pow(2, cellSizeExpLevel);

                    meshOffset += levelInfo[level - 1].CellCount;
                    _levelIdx[level] = meshOffset;

                    int levelOffset = 0;

                    if (level > 1)
                    {
                        levelOffset = _levelIdx[level - 1];
                        readOffset += levelInfo[level - 1].CellCount;
                    }

                    LevelInfo current = levelInfo[level];
                    int cellCount = current.CellCount;

                    _buildTopKernel.SetMemoryArgument(0, _gpuMesh);
                    _buildTopKernel.SetValueArgument(1, cellCount);
                    _buildTopKernel.SetMemoryArgument(2, gpuCellIdx);
                    _buildTopKernel.SetMemoryArgument(3, gpuLocIdx);
                    _buildTopKernel.SetValueArgument(4, cellSize);
                    _buildTopKernel.SetValueArgument(5, levelOffset);
                    _buildTopKernel.SetValueArgument(6, meshOffset);
                    _buildTopKernel.SetValueArgument(7, readOffset);
                    _buildTopKernel.SetValueArgument(8, level);

                    _queue.Execute(_buildTopKernel, null, new long[] { BlockCount(cellCount) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                }
            }
        }

        private void PopGridAndNeighborsGPU(GridInfo[] gridInfo, int meshSize)
        {
            // Calulate total size of 1D mesh neighbor index.
            // Each cell can have a max of 9 neighbors, including itself.
            int neighborLen = meshSize * 9;

            // Reallocate and resize GPU buffer as needed.
            Allocate(ref _gpuMeshNeighbors, neighborLen);

            // Calculate total size of 1D grid index.
            long gridSize = 0;
            foreach (var g in gridInfo)
            {
                gridSize += g.Size;
            }

            // Reallocate and resize GPU buffer as needed.
            long newCap = Allocate(ref _gpuGridIndex, gridSize);
            if (newCap > 0)
            {
                // Fill the new buffer with zeros.
                _queue.FillBuffer(_gpuGridIndex, new int[1] { 0 }, 0, newCap, null);
            }

            // Since the grid index can quickly exceed the maximum allocation size,
            // we do multiple passes with the same buffer, offsetting the bucket
            // indexes on each pass to fit within the buffer.

            // Calculate number of passes required to compute all neighbor cells.
            long passOffset = 0;
            long passes = 1;
            long stride = gridSize;
            long gridMem = gridSize * 4; // Size of grid index in memory. (n * bytes) (int = 4 bytes)

            // Do we need more than 1 pass?
            if (gridMem > _maxBufferSize)
            {
                passes += (gridMem / _maxBufferSize);
                stride = (int)_gpuGridIndex.Count;
            }

            for (int i = 0; i < passes; i++)
            {
                passOffset = stride * i;

                using (var gpuGridInfo = new ComputeBuffer<GridInfo>(_context, ComputeMemoryFlags.ReadOnly, gridInfo.Length, IntPtr.Zero))
                {
                    // Write Grid info to GPU.
                    _queue.WriteToBuffer(gridInfo, gpuGridInfo, false, null);

                    // Pop grid.
                    int argi = 0;
                    _popGridKernel.SetMemoryArgument(argi++, _gpuGridIndex);
                    _popGridKernel.SetValueArgument(argi++, (int)stride);
                    _popGridKernel.SetValueArgument(argi++, (int)passOffset);
                    _popGridKernel.SetMemoryArgument(argi++, gpuGridInfo);
                    _popGridKernel.SetMemoryArgument(argi++, _gpuMesh);
                    _popGridKernel.SetValueArgument(argi++, meshSize);

                    _queue.Execute(_popGridKernel, null, new long[] { BlockCount(meshSize) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);

                    // Build neighbor index.
                    argi = 0;
                    _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuMesh);
                    _buildNeighborsKernel.SetValueArgument(argi++, meshSize);
                    _buildNeighborsKernel.SetMemoryArgument(argi++, gpuGridInfo);
                    _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuGridIndex);
                    _buildNeighborsKernel.SetValueArgument(argi++, (int)stride);
                    _buildNeighborsKernel.SetValueArgument(argi++, (int)passOffset);
                    _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);

                    _queue.Execute(_buildNeighborsKernel, null, new long[] { BlockCount(meshSize) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);

                    // We're done with the grid index, so undo what we added to clear it for the next frame.
                    _clearGridKernel.SetMemoryArgument(0, _gpuGridIndex);
                    _clearGridKernel.SetValueArgument(1, (int)stride);
                    _clearGridKernel.SetValueArgument(2, (int)passOffset);
                    _clearGridKernel.SetMemoryArgument(3, _gpuMesh);
                    _clearGridKernel.SetValueArgument(4, meshSize);

                    _queue.Execute(_clearGridKernel, null, new long[] { BlockCount(meshSize) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                }

                //  _queue.Finish();
            }
        }

        /// <summary>
        /// Computes parameters for partitioning the specified length into the specified number of parts.
        /// </summary>
        /// <param name="length">Total number of items to be partitioned.</param>
        /// <param name="parts">Number of partitions to compute.</param>
        /// <param name="partLen">Computed length of each part.</param>
        /// <param name="modulo">Computed modulo or remainder to be added to the last partitions length.</param>
        /// <param name="count">Computed number of partitions. If parts is greater than length, this will be 1.</param>
        private void Partition(int length, int parts, out int partLen, out int modulo, out int count)
        {
            int outpLen, outMod;

            outpLen = length / parts;
            outMod = length % parts;

            if (parts >= length || outpLen <= 1)
            {
                partLen = length;
                modulo = 0;
                count = 1;
            }
            else
            {
                partLen = outpLen;
                modulo = outMod;
                count = parts;
            }

        }

        private long Allocate<T>(ref ComputeBuffer<T> buffer, long size, bool exactSize = false) where T : struct
        {
            long typeSize = Marshal.SizeOf<T>();
            long maxCap = (_maxBufferSize / typeSize);
            long handleVal = buffer.Handle.Value.ToInt64();

            BufferDims dims;

            if (!_bufferInfo.TryGetValue(handleVal, out dims))
            {
                dims = new BufferDims(handleVal, (int)size, (int)size, exactSize);
                _bufferInfo.Add(handleVal, dims);
            }

            var flags = buffer.Flags;

            if (!dims.ExactSize)
            {
                if (dims.Capacity < size && dims.Capacity < maxCap)
                {
                    buffer.Dispose();
                    _bufferInfo.Remove(handleVal);

                    long newCapacity = (long)(size * dims.GrowFactor);

                    // Clamp size to max allowed.
                    if (newCapacity > maxCap)
                    {
                        newCapacity = (int)maxCap;
                        size = newCapacity;
                    }

                    buffer = new ComputeBuffer<T>(_context, flags, newCapacity, IntPtr.Zero);

                    long newHandle = buffer.Handle.Value.ToInt64();
                    var newDims = new BufferDims(newHandle, (int)newCapacity, (int)size, exactSize);
                    _bufferInfo.Add(newHandle, newDims);

                    return newCapacity;
                }
            }
            else
            {
                if (dims.Size != size)
                {
                    buffer.Dispose();
                    _bufferInfo.Remove(handleVal);

                    // Clamp size to max allowed.
                    if (size * typeSize > _maxBufferSize)
                        size = maxCap;

                    buffer = new ComputeBuffer<T>(_context, flags, size, IntPtr.Zero);

                    long newHandle = buffer.Handle.Value.ToInt64();
                    var newDims = new BufferDims(newHandle, (int)size, (int)size, exactSize);
                    _bufferInfo.Add(newHandle, newDims);

                    return size;
                }
            }

            return 0;
        }

        private T[] ReadBuffer<T>(ComputeBuffer<T> buffer) where T : struct
        {
            T[] buf = new T[buffer.Count];

            _queue.ReadFromBuffer(buffer, ref buf, true, null);
            _queue.Finish();

            return buf;
        }

        private T[] ReadBuffer<T>(ComputeBuffer<T> buffer, long offset, long length) where T : struct
        {
            T[] buf = new T[length - offset];
            // T[] buf = new T[buffer.Count];


            _queue.ReadFromBuffer(buffer, ref buf, true, offset, 0, length - offset, null);
            _queue.Finish();

            return buf;
        }

        public void Dispose()
        {
            _gpuLevelIdx.Dispose();
            _gpuMesh.Dispose();
            _gpuMeshNeighbors.Dispose();
            _gpuInBodies.Dispose();
            _gpuOutBodies.Dispose();
            _gpuGridIndex.Dispose();

            _forceKernel.Dispose();
            _collisionKernel.Dispose();
            _popGridKernel.Dispose();
            _clearGridKernel.Dispose();
            _buildNeighborsKernel.Dispose();
            _fixOverlapKernel.Dispose();

            _program.Dispose();
            _context.Dispose();
            _queue.Dispose();
        }

        //private List<MeshCell> GetAllParents(Body body, MeshCell[] mesh)
        //{
        //    var parents = new List<MeshCell>();

        //    int parentId = body.MeshID;

        //    while (parentId != 0)
        //    {
        //        parents.Add(mesh[parentId]);

        //        parentId = mesh[parentId].ParentID;
        //    }

        //    return parents;
        //}
    }
}