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
        private int _gpuIndex = 2;
        private int _levels = 4;
        private static int _threadsPerBlock = 256;
        private int _parts = 24;

        private int[] _levelIdx = new int[0];
        private MeshCell[] _mesh = new MeshCell[0];
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
        private ComputeKernel _popGridKernel;
        private ComputeKernel _clearGridKernel;
        private ComputeKernel _buildNeighborsKernel;
        private ComputeKernel _fixOverlapKernel;

        private ComputeBuffer<int> _gpuLevelIdx;
        private ComputeBuffer<MeshCell> _gpuMesh;
        private ComputeBuffer<int> _gpuMeshNeighbors;
        private ComputeBuffer<Body> _gpuInBodies;
        private ComputeBuffer<Body> _gpuOutBodies;
        private ComputeBuffer<int> _gpuGridIndex;

        private static Dictionary<long, BufferDims> _bufferInfo = new Dictionary<long, BufferDims>();
        private static ParallelOptions _parallelOptions = new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount };

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
            _popGridKernel = _program.CreateKernel("PopGrid");
            _buildNeighborsKernel = _program.CreateKernel("BuildNeighbors");
            _clearGridKernel = _program.CreateKernel("ClearGrid");
            _fixOverlapKernel = _program.CreateKernel("FixOverlaps");

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

        public void CalcMovement(ref Body[] bodies, float timestep, float viscosity, int cellSizeExp, float cullDistance, int meshLevels, int threadsPerBlock)
        {
            _bodies = bodies;
            _threadsPerBlock = threadsPerBlock;
            _levels = meshLevels;
            int threadBlocks = 0;

            // Calc number of thread blocks to fit the dataset.
            threadBlocks = BlockCount(_bodies.Length);

            // Build the particle mesh, mesh index, and mesh neighbors index.
            BuildMesh(cellSizeExp);

            var centerMass = CalcCenterMass();

            Allocate(ref _gpuLevelIdx, _levelIdx.Length, true);
            _queue.WriteToBuffer(_levelIdx, _gpuLevelIdx, true, null);
            _queue.Finish();

            int argi = 0;
            _forceKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _forceKernel.SetValueArgument(argi++, _bodies.Length);
            _forceKernel.SetMemoryArgument(argi++, _gpuOutBodies);
            _forceKernel.SetMemoryArgument(argi++, _gpuMesh);
            _forceKernel.SetValueArgument(argi++, _mesh.Length);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _forceKernel.SetValueArgument(argi++, timestep);
            _forceKernel.SetValueArgument(argi++, _levels);
            _forceKernel.SetMemoryArgument(argi++, _gpuLevelIdx);
            _forceKernel.SetValueArgument(argi++, _levelIdx.Length);

            _queue.Execute(_forceKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);
            _queue.Finish();


            argi = 0;
            _collisionKernel.SetMemoryArgument(argi++, _gpuOutBodies);
            _collisionKernel.SetValueArgument(argi++, _bodies.Length);
            _collisionKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _collisionKernel.SetMemoryArgument(argi++, _gpuMesh);
            _collisionKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _collisionKernel.SetValueArgument(argi++, timestep);
            _collisionKernel.SetValueArgument(argi++, viscosity);
            _collisionKernel.SetValueArgument(argi++, centerMass);
            _collisionKernel.SetValueArgument(argi++, cullDistance);

            _queue.Execute(_collisionKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);
            _queue.Finish();

            _queue.ReadFromBuffer(_gpuInBodies, ref bodies, true, null);
            _queue.Finish();
        }

        private Vector2 CalcCenterMass()
        {
            double cmX = 0;
            double cmY = 0;
            double mass = 0;

            for (int i = _levelIdx[_levels]; i < _mesh.Length; i++)
            {
                var cell = _mesh[i];

                mass += cell.Mass;
                cmX += cell.Mass * cell.CmX;
                cmY += cell.Mass * cell.CmY;
            }

            cmX = cmX / mass;
            cmY = cmY / mass;

            return new Vector2((float)cmX, (float)cmY);
        }

        private void WriteBodiesToGPU()
        {
            Allocate(ref _gpuInBodies, _bodies.Length, true);
            Allocate(ref _gpuOutBodies, _bodies.Length, true);

            _queue.WriteToBuffer(_bodies, _gpuInBodies, false, null);
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

        // Calculate dimensionless morton number from X/Y coords.
        private static readonly int[] B = new int[] { 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF };
        private static readonly int[] S = new int[] { 1, 2, 4, 8 };

        private int MortonNumber(int x, int y)
        {
            x &= 0x0000ffff;
            x = (x | (x << S[3])) & B[3];
            x = (x | (x << S[2])) & B[2];
            x = (x | (x << S[1])) & B[1];
            x = (x | (x << S[0])) & B[0];

            y &= 0x0000ffff;
            y = (y | (y << S[3])) & B[3];
            y = (y | (y << S[2])) & B[2];
            y = (y | (y << S[1])) & B[1];
            y = (y | (y << S[0])) & B[0];

            return x | (y << 1);
        }

        private void AddGridDims(MinMax minMax, int level)
        {
            int minXAbs = Math.Abs(minMax.MinX - 1);
            int minYAbs = Math.Abs(minMax.MinY - 1);

            int columns = minXAbs + minMax.MaxX;
            int rows = minYAbs + minMax.MaxY;

            int idxOff = 0;
            int size = ((columns + 1) * (rows + 1));

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

            _gridInfo[level] = new GridInfo(minXAbs, minYAbs, idxOff, minMax.MinX, minMax.MinY, minMax.MaxX, minMax.MaxY, columns, rows);

        }

        /// <summary>
        /// Computes spatial info (Morton number, X/Y indexes, mesh cell count) for all bodies.
        /// </summary>
        /// <param name="cellSizeExp">Cell size exponent. "Math.Pow(2, exponent)"</param>
        private LevelInfo CalcBodySpatials(int cellSizeExp)
        {
            if (_spatials.Length != _bodies.Length)
                _spatials = new SpatialInfo[_bodies.Length];

            // Array of morton numbers used for sorting.
            // Using a key array when sorting is much faster than sorting an array of objects by a field.
            if (_mortKeys.Length != _bodies.Length)
                _mortKeys = new int[_bodies.Length];

            int pLen, pRem, pCount;
            Partition(_bodies.Length, _parts, out pLen, out pRem, out pCount);

            // Compute the spatial info in parallel.
            Parallel.For(0, pCount, (p) =>
            {
                int offset = p * pLen;
                int len = offset + pLen;

                if (p == pCount - 1)
                    len += pRem;

                for (int b = offset; b < len; b++)
                {
                    int idxX = (int)_bodies[b].PosX >> cellSizeExp;
                    int idxY = (int)_bodies[b].PosY >> cellSizeExp;
                    int morton = MortonNumber(idxX, idxY);

                    _spatials[b] = new SpatialInfo(morton, idxX, idxY, b);
                    _mortKeys[b] = morton;
                }

            });

            // Sort by morton number to produce a spatially sorted array.
            Array.Sort(_mortKeys, _spatials);

            // Compute number of unique morton numbers to determine cell count,
            // and build the start index of each cell.
            int count = 0;
            int val = 0;

            if (_cellIdx.Length != _bodies.Length + 1)
                _cellIdx = new int[_bodies.Length + 1];

            if (_sortBodies.Length != _bodies.Length)
                _sortBodies = new Body[_bodies.Length];


            for (int i = 0; i < _spatials.Length; i++)
            {
                // Build a new sorted body array from the sorted spatial info.
                _sortBodies[i] = _bodies[_spatials[i].Index];

                // Update spatials index to match new sorted bodies.
                _spatials[i].Index = i;

                if (val != _spatials[i].Mort)
                {
                    _cellIdx[count] = i;
                    val = _spatials[i].Mort;

                    count++;
                }
            }

            _cellIdx[count] = _spatials.Length;

            // Update the original body array with the sorted one.
            _bodies = _sortBodies;

            var output = new LevelInfo();
            output.Spatials = _spatials;
            output.CellCount = count;
            output.CellIndex = new int[count + 1];
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
                LevelInfo current = output[level - 1];

                output[level] = new LevelInfo();
                output[level].Spatials = new SpatialInfo[current.CellCount];

                int pLen, pRem, pCount;
                Partition(current.CellCount, _parts, out pLen, out pRem, out pCount);

                Parallel.For(0, pCount, (p) =>
                {
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

                        output[level].Spatials[b] = new SpatialInfo(morton, idxX, idxY, spatial.Index + b);
                    }

                });

                int count = 0;
                int val = int.MaxValue;

                for (int i = 0; i < output[level].Spatials.Length; i++)
                {
                    if (val != output[level].Spatials[i].Mort)
                    {
                        _cellIdx[count] = i;
                        val = output[level].Spatials[i].Mort;

                        count++;

                    }
                }

                _cellIdx[count] = output[level].Spatials.Length;

                output[level].CellCount = count;
                output[level].CellIndex = new int[count + 1];
                Array.Copy(_cellIdx, 0, output[level].CellIndex, 0, count + 1);
            }

            return output;
        }

        /// <summary>
        /// Builds the particle mesh and mesh-neighbor index for the current field.  Also begins writing the body array to GPU...
        /// </summary>
        /// <param name="cellSizeExp">Cell size exponent. 2 ^ exponent = cell size.</param>
        private void BuildMesh(int cellSizeExp)
        {
            // Get spatial info for the cells about to be constructed.
            LevelInfo botLevel = CalcBodySpatials(cellSizeExp);
            LevelInfo[] topLevels = CalcTopSpatials(botLevel);

            int totCells = 0;
            foreach (var lvl in topLevels)
            {
                totCells += lvl.CellCount;
            }

            if (_mesh.Length != totCells)
                _mesh = new MeshCell[totCells];

            _gridInfo = new GridInfo[_levels + 1];
            object lockObj = new object();
            MinMax minMax = new MinMax();

            int cellCount = botLevel.CellCount;
            int[] cellStartIdx = botLevel.CellIndex;
            int cellSize = (int)Math.Pow(2, cellSizeExp);

            int pLen, pRem, pCount;
            Partition(cellCount, _parts, out pLen, out pRem, out pCount);

            // Use the spatial info to quickly construct the first level of mesh cells in parallel.
            Parallel.For(0, pCount, (p) =>
            {
                var mm = new MinMax();

                int offset = p * pLen;
                int len = offset + pLen;

                if (p == pCount - 1)
                    len += pRem;

                for (int m = offset; m < len; m++)
                {
                    // Get the spatial info from the first cell index; there may only be one cell.
                    var spatial = botLevel.Spatials[cellStartIdx[m]];

                    var newCell = new MeshCell();
                    newCell.LocX = (spatial.IdxX << cellSizeExp) + (cellSize * 0.5f);
                    newCell.LocY = (spatial.IdxY << cellSizeExp) + (cellSize * 0.5f);
                    newCell.IdxX = spatial.IdxX;
                    newCell.IdxY = spatial.IdxY;
                    newCell.Size = cellSize;
                    newCell.Mort = spatial.Mort;

                    mm.Update(spatial.IdxX, spatial.IdxY);

                    // Iterate the elements between the spatial info cell indexes and add body info.
                    for (int i = cellStartIdx[m]; i < cellStartIdx[m + 1]; i++)
                    {
                        var body = _bodies[i];
                        newCell.Mass += body.Mass;
                        newCell.CmX += body.Mass * body.PosX;
                        newCell.CmY += body.Mass * body.PosY;
                        newCell.BodyCount++;

                        _bodies[i].MeshID = m;
                    }

                    newCell.ID = m;
                    newCell.BodyStartIdx = cellStartIdx[m];

                    newCell.CmX = newCell.CmX / (float)newCell.Mass;
                    newCell.CmY = newCell.CmY / (float)newCell.Mass;

                    _mesh[m] = newCell;
                }

                lock (lockObj)
                {
                    minMax.Update(mm);
                }

            });

            AddGridDims(minMax, 0);

            // At this point we are done modifying the body array,
            // so go ahead and start writing it to the GPU with a non-blocking call.
            WriteBodiesToGPU();

            // Index to hold the starting indexes for each level.
            _levelIdx = new int[_levels + 1];
            _levelIdx[0] = 0;

            BuildTopLevels(topLevels, cellSizeExp, _levels);

            // Allocate and begin writing mesh to GPU.
            Allocate(ref _gpuMesh, _mesh.Length, true);
            _queue.WriteToBuffer(_mesh, _gpuMesh, false, null);

            // Populate the grid index and calulate mesh neighbor index.
            PopGridAndNeighborsGPU(_gridInfo, _mesh.Length);
        }

        private void BuildTopLevels(LevelInfo[] levelInfo, int cellSizeExp, int levels)
        {
            int meshOffset = 0;
            for (int level = 1; level <= levels; level++)
            {
                int cellSizeExpLevel = cellSizeExp + level;
                int cellSize = (int)Math.Pow(2, cellSizeExpLevel);

                meshOffset += levelInfo[level - 1].CellCount;
                _levelIdx[level] = meshOffset;

                int levelOffset = 0;

                if (level > 1)
                    levelOffset = _levelIdx[level - 1];

                LevelInfo current = levelInfo[level];
                int cellCount = current.CellCount;
                object lockObj = new object();
                MinMax minMax = new MinMax();

                int pLen, pRem, pCount;
                Partition(cellCount, _parts, out pLen, out pRem, out pCount);

                Parallel.For(0, pCount, (p) =>
                {
                    var mm = new MinMax();

                    int offset = p * pLen;
                    int len = offset + pLen;

                    if (p == pCount - 1)
                        len += pRem;

                    for (int m = offset; m < len; m++)
                    {
                        // Get the spatial info from the first cell index; there may only be one cell.
                        var spatial = current.Spatials[current.CellIndex[m]];
                        int newIdx = m + meshOffset;

                        var newCell = new MeshCell();
                        newCell.LocX = (spatial.IdxX << cellSizeExpLevel) + (cellSize * 0.5f);
                        newCell.LocY = (spatial.IdxY << cellSizeExpLevel) + (cellSize * 0.5f);
                        newCell.IdxX = spatial.IdxX;
                        newCell.IdxY = spatial.IdxY;
                        newCell.Size = cellSize;
                        newCell.Mort = spatial.Mort;
                        newCell.ChildStartIdx = current.CellIndex[m] + levelOffset;
                        newCell.ChildCount = 0;
                        newCell.ID = newIdx;
                        newCell.Level = level;
                        newCell.BodyStartIdx = 0;
                        newCell.BodyCount = 0;

                        mm.Update(spatial.IdxX, spatial.IdxY);

                        // Iterate the elements between the spatial info cell indexes and add child info.
                        for (int i = current.CellIndex[m]; i < current.CellIndex[m + 1]; i++)
                        {
                            var child = _mesh[i + levelOffset];
                            newCell.Mass += child.Mass;
                            newCell.CmX += (float)child.Mass * child.CmX;
                            newCell.CmY += (float)child.Mass * child.CmY;
                            newCell.BodyCount += child.BodyCount;
                            newCell.ChildCount++;
                            child.ParentID = newIdx;
                            _mesh[i + levelOffset] = child;
                        }

                        newCell.CmX = newCell.CmX / (float)newCell.Mass;
                        newCell.CmY = newCell.CmY / (float)newCell.Mass;

                        _mesh[newIdx] = newCell;
                    }

                    lock (lockObj)
                    {
                        minMax.Update(mm);
                    }

                });

                AddGridDims(minMax, level);
            }
        }

        private void PopGridAndNeighborsGPU(GridInfo[] gridInfo, int meshSize)
        {
            // Calculate total size of 1D grid index.
            int gridSize = 0;
            foreach (var g in gridInfo)
            {
                gridSize += g.Size;
            }

            // Reallocate and resize GPU buffer as needed.
            int newCap = Allocate(ref _gpuGridIndex, gridSize);
            if (newCap > 0)
            {
                _queue.FillBuffer<int>(_gpuGridIndex, new int[1] { 0 }, 0, newCap, null);
                _queue.Finish();
            }

            // Calulate total size of 1D mesh neighbor index.
            int neighborLen = meshSize * 9;

            // Reallocate and resize GPU buffer as needed.
            Allocate(ref _gpuMeshNeighbors, neighborLen);

            using (var gpuGridInfo = new ComputeBuffer<GridInfo>(_context, ComputeMemoryFlags.ReadOnly, gridInfo.Length, IntPtr.Zero))
            {
                // Write Grid info.
                _queue.WriteToBuffer(gridInfo, gpuGridInfo, true, null);
                _queue.Finish();

                // Pop grid.
                int argi = 0;
                _popGridKernel.SetMemoryArgument(argi++, _gpuGridIndex);
                _popGridKernel.SetMemoryArgument(argi++, gpuGridInfo);
                _popGridKernel.SetMemoryArgument(argi++, _gpuMesh);
                _popGridKernel.SetValueArgument(argi++, meshSize);

                _queue.Execute(_popGridKernel, null, new long[] { BlockCount(meshSize) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                _queue.Finish();

                // Build neighbor index.
                argi = 0;
                _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuMesh);
                _buildNeighborsKernel.SetValueArgument(argi++, meshSize);
                _buildNeighborsKernel.SetMemoryArgument(argi++, gpuGridInfo);
                _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuGridIndex);
                _buildNeighborsKernel.SetValueArgument(argi++, gridSize);
                _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);

                _queue.Execute(_buildNeighborsKernel, null, new long[] { BlockCount(meshSize) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                _queue.Finish();

                // We're done with the grid index, so undo what we added to clear it for the next frame.
                _clearGridKernel.SetMemoryArgument(0, _gpuGridIndex);
                _clearGridKernel.SetMemoryArgument(1, _gpuMesh);
                _clearGridKernel.SetValueArgument(2, meshSize);

                _queue.Execute(_clearGridKernel, null, new long[] { BlockCount(meshSize) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                _queue.Finish();
            }
        }

        private void Partition(int length, int parts, out int pLen, out int rem, out int count)
        {
            int outpLen, outRem;

            outpLen = length / parts;
            outRem = length % parts;

            if (parts >= length || outpLen <= 1)
            {
                pLen = length;
                rem = 0;
                count = 1;
            }
            else
            {
                pLen = outpLen;
                rem = outRem;
                count = parts;
            }

        }

        private int Allocate<T>(ref ComputeBuffer<T> buffer, int size, bool exactSize = false) where T : struct
        {
            long handleVal = buffer.Handle.Value.ToInt64();

            if (!_bufferInfo.ContainsKey(handleVal))
            {
                _bufferInfo.Add(handleVal, new BufferDims(handleVal, size, size, exactSize));
            }

            var dims = _bufferInfo[handleVal];
            var flags = buffer.Flags;

            if (!dims.ExactSize)
            {
                if (dims.Capacity < size)
                {
                    buffer.Dispose();
                    _bufferInfo.Remove(handleVal);

                    int newCapacity = (int)(size * dims.GrowFactor);

                    buffer = new ComputeBuffer<T>(_context, flags, newCapacity, IntPtr.Zero);

                    long newHandle = buffer.Handle.Value.ToInt64();
                    var newDims = new BufferDims(newHandle, newCapacity, size, exactSize);
                    _bufferInfo.Add(newHandle, newDims);

                    //Console.WriteLine($@"[{newDims.Name}]  {newDims.Capacity}");

                    return newCapacity;
                }
            }
            else
            {
                if (dims.Size != size)
                {
                    buffer.Dispose();
                    _bufferInfo.Remove(handleVal);

                    buffer = new ComputeBuffer<T>(_context, flags, size, IntPtr.Zero);

                    long newHandle = buffer.Handle.Value.ToInt64();
                    var newDims = new BufferDims(newHandle, size, size, exactSize);
                    _bufferInfo.Add(newHandle, newDims);

                    //Console.WriteLine($@"[{newDims.Name}]  {newDims.Capacity}");

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