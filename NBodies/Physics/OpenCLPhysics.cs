using Cloo;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Threading;
using System.Runtime.CompilerServices;
using NBodies.Extensions;

namespace NBodies.Physics
{
    public class OpenCLPhysics : IPhysicsCalc, IDisposable
    {
        private int _gpuIndex = 4;
        private int _levels = 4;
        private static int _threadsPerBlock = 256;
        private int _parallelPartitions = 12;//24;
        private long _maxBufferSize = 0;
        private float _kernelSize = 1.0f;
        private SPHPreCalc _preCalcs;

        private int[] _levelIdx = new int[0]; // Locations of each level within the 1D mesh array.
        private MeshCell[] _mesh = new MeshCell[0]; // 1D array of mesh cells. (Populated on GPU, and read for UI display only.)
        private int _meshLength = 0; // Total length of the 1D mesh array.
        private GridInfo[] _gridInfo = new GridInfo[0]; // Bounds and offsets for each level of the neighbor list grid.
        private LevelInfo[] _levelInfo = new LevelInfo[0]; // Spatial info, cell index map and counts for each level.
        private Body[] _bodies = new Body[0]; // Local reference for the current body array.

        // Static fields for large allocations.
        private static SpatialInfo[] _bSpatials = new SpatialInfo[0]; // Body spatials.
        private static int[] _mortKeys = new int[0]; // Array of Morton numbers for spatials sorting.
        private static int[] _allCellIdx = new int[0]; // Buffer for completed/flattened cell index map.

        private ManualResetEventSlim _meshReadyWait = new ManualResetEventSlim(false);

        private ComputeContext _context;
        private ComputeCommandQueue _queue;
        private ComputeDevice _device = null;
        private ComputeProgram _program;

        private ComputeKernel _forceKernel;
        private ComputeKernel _collisionSPHKernel;
        private ComputeKernel _collisionElasticKernel;
        private ComputeKernel _popGridKernel;
        private ComputeKernel _clearGridKernel;
        private ComputeKernel _buildNeighborsKernel;
        private ComputeKernel _fixOverlapKernel;
        private ComputeKernel _buildBottomKernel;
        private ComputeKernel _buildTopKernel;
        private ComputeKernel _calcCMKernel;
        private ComputeKernel _reindexKernel;

        private ComputeBuffer<MeshCell> _gpuMesh;
        private ComputeBuffer<int> _gpuMeshNeighbors;
        private ComputeBuffer<Body> _gpuInBodies;
        private ComputeBuffer<Body> _gpuOutBodies;
        private ComputeBuffer<int> _gpuGridIndex;
        private ComputeBuffer<Vector2> _gpuCM;
        private ComputeBuffer<int> _gpuSortMap;
        private ComputeBuffer<int> _gpuCellIdx;
        private ComputeBuffer<GridInfo> _gpuGridInfo;
        private ComputeBuffer<int> _gpuPostNeeded;

        private static Dictionary<long, BufferDims> _bufferInfo = new Dictionary<long, BufferDims>();

        private Stopwatch timer = new Stopwatch();
        private Stopwatch timer2 = new Stopwatch();


        public MeshCell[] CurrentMesh
        {
            get
            {
                // Only read mesh from GPU if it has changed.
                if (_mesh.Length != _meshLength)
                {
                    // Wait for build to finish.
                    _meshReadyWait.Wait();
                    _mesh = new MeshCell[_meshLength];
                    _queue.ReadFromBuffer(_gpuMesh, ref _mesh, false, 0, 0, _meshLength, null);
                    _queue.Finish();
                }

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

        public OpenCLPhysics(ComputeDevice device, int threadsperblock)
        {
            _device = device;

            if (threadsperblock != -1)
                _threadsPerBlock = threadsperblock;
        }

        public void Init()
        {
            var devices = GetDevices();

            if (_device == null)
                _device = devices[_gpuIndex];

            var platform = _device.Platform;

            _maxBufferSize = _device.MaxMemoryAllocationSize;
            _context = new ComputeContext(new[] { _device }, new ComputeContextPropertyList(platform), null, IntPtr.Zero);
            _queue = new ComputeCommandQueue(_context, _device, ComputeCommandQueueFlags.None);

            StreamReader streamReader = new StreamReader(Environment.CurrentDirectory + "/Physics/Kernels.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            _program = new ComputeProgram(_context, clSource);

            try
            {
                _program.Build(null, "-cl-std=CL1.2 -cl-fast-relaxed-math", null, IntPtr.Zero);
            }
            catch (BuildProgramFailureComputeException ex)
            {
                string buildLog = _program.GetBuildLog(_device);
                System.IO.File.WriteAllText("build_error.txt", buildLog);
                Console.WriteLine(buildLog);
                throw;
            }


            Console.WriteLine(_program.GetBuildLog(_device));
            System.IO.File.WriteAllText("build_log.txt", _program.GetBuildLog(_device));

            _forceKernel = _program.CreateKernel("CalcForce");
            _collisionSPHKernel = _program.CreateKernel("SPHCollisions");
            _collisionElasticKernel = _program.CreateKernel("ElasticCollisions");
            _popGridKernel = _program.CreateKernel("PopGrid");
            _buildNeighborsKernel = _program.CreateKernel("BuildNeighbors");
            _clearGridKernel = _program.CreateKernel("ClearGrid");
            _fixOverlapKernel = _program.CreateKernel("FixOverlaps");
            _buildBottomKernel = _program.CreateKernel("BuildBottom");
            _buildTopKernel = _program.CreateKernel("BuildTop");
            _calcCMKernel = _program.CreateKernel("CalcCenterOfMass");
            _reindexKernel = _program.CreateKernel("ReindexBodies");

            InitBuffers();

            PreCalcSPH(_kernelSize);
        }

        public static List<ComputeDevice> GetDevices()
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

        private void PreCalcSPH(float kernelSize)
        {
            var calcs = new SPHPreCalc();

            calcs.kSize = kernelSize;
            calcs.kSizeSq = (float)Math.Pow(kernelSize, 2);
            calcs.kSize3 = (float)Math.Pow(kernelSize, 3);
            calcs.kSize9 = (float)Math.Pow(kernelSize, 9);
            calcs.kRad6 = (float)Math.Pow((1.0f / 3.0f), 6); //??
            // calcs.kRad6 = (float)Math.Pow(kernelSize, 6);

            calcs.fViscosity = (float)(15.0f / (2.0f * Math.PI * calcs.kSize3)) * (6.0f / calcs.kSize3);
            calcs.fPressure = (float)(15.0f / (Math.PI * calcs.kRad6)) * 3.0f;
            calcs.fDensity = (float)(315.0f / (64.0f * Math.PI * calcs.kSize9));

            _preCalcs = calcs;
        }

        public void Flush()
        {
            _meshReadyWait.Reset();
            _meshLength = 0;
            _gpuMesh.Dispose();
            _gpuMeshNeighbors.Dispose();
            _gpuInBodies.Dispose();
            _gpuOutBodies.Dispose();
            _gpuGridIndex.Dispose();
            _gpuCM.Dispose();
            _gpuSortMap.Dispose();
            _gpuCellIdx.Dispose();
            _gpuGridInfo.Dispose();
            _gpuPostNeeded.Dispose();
            _mesh = new MeshCell[0];
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

            _gpuSortMap = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadOnly, 1, IntPtr.Zero);
            Allocate(ref _gpuSortMap, 0, true);

            _gpuCellIdx = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadOnly, 1, IntPtr.Zero);
            Allocate(ref _gpuCellIdx, 0, true);

            _gpuGridInfo = new ComputeBuffer<GridInfo>(_context, ComputeMemoryFlags.ReadOnly, 1, IntPtr.Zero);
            Allocate(ref _gpuGridInfo, 0, true);

            _gpuPostNeeded = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1, IntPtr.Zero);
            Allocate(ref _gpuPostNeeded, 1, true);

            _gpuCM = new ComputeBuffer<Vector2>(_context, ComputeMemoryFlags.ReadWrite, 1, IntPtr.Zero);
            Allocate(ref _gpuCM, 1, true);
        }

        public void CalcMovement(ref Body[] bodies, SimSettings sim, int threadsPerBlock, out bool isPostNeeded)
        {
            _bodies = bodies;
            _threadsPerBlock = threadsPerBlock;
            _levels = sim.MeshLevels;
            int threadBlocks = 0;

            // Allocate and start writing bodies to the GPU.
            Allocate(ref _gpuInBodies, _bodies.Length);
            Allocate(ref _gpuOutBodies, _bodies.Length);
            _queue.WriteToBuffer(_bodies, _gpuOutBodies, false, 0, 0, _bodies.Length, null);

            if (_kernelSize != sim.KernelSize)
            {
                _kernelSize = sim.KernelSize;
                PreCalcSPH(_kernelSize);
            }

            // Calc number of thread blocks to fit the dataset.
            threadBlocks = BlockCount(_bodies.Length);

            // Block mesh reads until it's finised building.
            _meshReadyWait.Reset();

            // Build the particle mesh, mesh index, and mesh neighbors index.
            BuildMesh(sim.CellSizeExponent);

            // Allow mesh read.
            _meshReadyWait.Set();

            // Calc center of mass on GPU from top-most level.
            _calcCMKernel.SetMemoryArgument(0, _gpuMesh);
            _calcCMKernel.SetMemoryArgument(1, _gpuCM);
            _calcCMKernel.SetValueArgument(2, _levelIdx[_levels]);
            _calcCMKernel.SetValueArgument(3, _meshLength);
            _queue.ExecuteTask(_calcCMKernel, null);

            int[] postNeeded = new int[1] { 0 };
            _queue.WriteToBuffer(postNeeded, _gpuPostNeeded, false, null);

            int argi = 0;
            _forceKernel.SetMemoryArgument(argi++, _gpuOutBodies);
            _forceKernel.SetValueArgument(argi++, _bodies.Length);
            _forceKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _forceKernel.SetMemoryArgument(argi++, _gpuMesh);
            _forceKernel.SetValueArgument(argi++, _levelIdx[_levels]);
            _forceKernel.SetValueArgument(argi++, _meshLength);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _forceKernel.SetValueArgument(argi++, sim);
            _forceKernel.SetValueArgument(argi++, _preCalcs);
            _forceKernel.SetMemoryArgument(argi++, _gpuPostNeeded);
            _queue.Execute(_forceKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);

            argi = 0;
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _collisionElasticKernel.SetValueArgument(argi++, _bodies.Length);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMesh);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _collisionElasticKernel.SetValueArgument(argi++, Convert.ToInt32(sim.CollisionsOn));
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuPostNeeded);
            _queue.Execute(_collisionElasticKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);

            argi = 0;
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _collisionSPHKernel.SetValueArgument(argi++, _bodies.Length);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuOutBodies);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMesh);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuCM);
            _collisionSPHKernel.SetValueArgument(argi++, sim);
            _collisionSPHKernel.SetValueArgument(argi++, _preCalcs);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuPostNeeded);
            _queue.Execute(_collisionSPHKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);

            isPostNeeded = Convert.ToBoolean(ReadBuffer(_gpuPostNeeded)[0]);

            _queue.ReadFromBuffer(_gpuOutBodies, ref bodies, true, 0, 0, bodies.Length, null);
            _queue.Finish();
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

                bodies = ReadBuffer(outBodies, true);
            }
        }

        /// <summary>
        /// Calculates number of thread blocks needed to fit the specified data length and the specified number of threads per block.
        /// </summary>
        /// <param name="len">Length of data set.</param>
        /// <param name="threads">Number of threads per block.</param>
        /// <returns>Number of blocks.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int BlockCount(int len, int threads = 0)
        {
            if (threads == 0)
                threads = _threadsPerBlock;

            int blocks = len / threads;
            int mod = len % threads;

            if (mod > 0)
                blocks += 1;

            return blocks;
        }

        /// <summary>
        /// Calculate dimensionless morton number from X/Y coords.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

            _gridInfo[level].Set(offsetX, offsetY, idxOff, minMax.MinX, minMax.MinY, minMax.MaxX, minMax.MaxY, columns, rows);
        }

        /// <summary>
        /// Builds the particle mesh and mesh-neighbor index for the current field.  Also begins writing the body array to GPU...
        /// </summary>
        /// <param name="cellSizeExp">Cell size exponent. 2 ^ exponent = cell size.</param>
        private void BuildMesh(int cellSizeExp)
        {
            // Grid info for each level.
            if (_gridInfo.Length != (_levels + 1))
                _gridInfo = new GridInfo[_levels + 1];

            // Mesh info for each level.
            if (_levelInfo.Length != (_levels + 1))
                _levelInfo = new LevelInfo[_levels + 1];

            // Compute cell count and cell index maps.
            // Also does spatial sorting of bodies.
            CalcBodySpatialsAndSort(cellSizeExp);
            CalcTopSpatials();

            // Get the total number of mesh cells to be created.
            int totCells = 0;
            foreach (var lvl in _levelInfo)
            {
                totCells += lvl.CellCount;
            }

            _meshLength = totCells;

            // Allocate the mesh buffer on the GPU.
            Allocate(ref _gpuMesh, totCells, false);

            // Index to hold the starting indexes for each level within the 1D mesh array.
            _levelIdx = new int[_levels + 1];
            _levelIdx[0] = 0;

            // Write the cell index map as one large array to the GPU.
            WriteCellIndex(_levelInfo);

            // Build the first (bottom) level of the mesh.
            BuildBottomLevelGPU(_levelInfo[0], cellSizeExp);

            // Build the remaining (top) levels of the mesh.
            BuildTopLevelsGPU(_levelInfo, cellSizeExp, _levels);

            // Populate the grid index and mesh neighbor index.
            PopGridAndNeighborsGPU(_gridInfo, totCells);
        }

        /// <summary>
        /// Computes spatial info (Morton number, X/Y indexes, mesh cell count) for all bodies.
        /// </summary>
        /// <param name="cellSizeExp">Cell size exponent. "Math.Pow(2, exponent)"</param>
        private unsafe void CalcBodySpatialsAndSort(int cellSizeExp)
        {
            // Spatial info to be computed.
            if (_bSpatials.Length < _bodies.Length)
                _bSpatials = new SpatialInfo[_bodies.Length];

            // Array of morton numbers used for sorting.
            // Using a key array when sorting is much faster than sorting an array of objects by a field.
            if (_mortKeys.Length < _bodies.Length)
                _mortKeys = new int[_bodies.Length];

            var minMax = new MinMax(0);
            var sync = new object();

            // Compute the spatial info in parallel.
            ParallelForSlim(_bodies.Length, _parallelPartitions, (start, len) =>
            {
                var mm = new MinMax(0);

                for (int b = start; b < len; b++)
                {
                    int idxX = (int)Math.Floor(_bodies[b].PosX) >> cellSizeExp;
                    int idxY = (int)Math.Floor(_bodies[b].PosY) >> cellSizeExp;
                    int morton = MortonNumber(idxX, idxY);

                    mm.Update(idxX, idxY);

                    _bSpatials[b].Set(morton, idxX, idxY, b);
                    _mortKeys[b] = morton;
                }

                lock (sync)
                {
                    minMax.Update(mm);
                }

            });

            AddGridDims(minMax, 0);

            // Sort by morton number to produce a spatially sorted array.
            // Array.Sort(_mortKeys, _spatials);
            Sort.ParallelQuickSort(_mortKeys, _bSpatials, _bodies.Length);

            // Compute number of unique morton numbers to determine cell count,
            // and build the start index of each cell.
            int count = 0;
            int val = int.MaxValue;

            var cellIdx = _levelInfo[0].CellIndex;

            if (cellIdx == null || cellIdx.Length < _bodies.Length)
                cellIdx = new int[_bodies.Length + 100];

            // Allocate then map the sort map buffer so we can write to it directly.
            Allocate(ref _gpuSortMap, _bodies.Length + 100);
            var sortMapPtr = _queue.Map(_gpuSortMap, true, ComputeMemoryMappingFlags.Write, 0, _gpuSortMap.Count, null);
            var sortMapNativePtr = (int*)sortMapPtr.ToPointer();

            fixed (int* mortPtr = _mortKeys, cellIdxPtr = cellIdx)
            fixed (SpatialInfo* spaPtr = _bSpatials)
            {
                for (int i = 0; i < _bodies.Length; i++)
                {
                    var mort = mortPtr[i];
                    sortMapNativePtr[i] = spaPtr[i].Index;

                    // Find the start of each new morton number and record location to build cell index.
                    if (val != mort)
                    {
                        cellIdxPtr[count] = i;
                        val = mort;
                        count++;
                    }
                }
            }

            // Add the last cell index value;
            cellIdx[count] = _bodies.Length;

            // Unmap the sort map buffer.
            _queue.Unmap(_gpuSortMap, ref sortMapPtr, null);

            // Start reindexing bodies on the GPU.
            ReindexBodiesGPU();

            _levelInfo[0].CellIndex = cellIdx;
            _levelInfo[0].Spatials = _bSpatials;
            _levelInfo[0].CellCount = count;
        }

        private void ReindexBodiesGPU()
        {
            _reindexKernel.SetMemoryArgument(0, _gpuOutBodies);
            _reindexKernel.SetValueArgument(1, _bodies.Length);
            _reindexKernel.SetMemoryArgument(2, _gpuSortMap);
            _reindexKernel.SetMemoryArgument(3, _gpuInBodies);
            _queue.Execute(_reindexKernel, null, new long[] { BlockCount(_bodies.Length) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
        }

        /// <summary>
        /// Computes spatial info (Morton number, X/Y indexes, mesh cell count) for all top mesh levels.
        /// </summary>
        private unsafe void CalcTopSpatials()
        {
            object sync = new object();
            MinMax minMax = new MinMax(0);

            for (int level = 1; level <= _levels; level++)
            {
                minMax.Reset();

                LevelInfo childLevel = _levelInfo[level - 1];
                int childCount = childLevel.CellCount;

                if (_levelInfo[level].Spatials == null || _levelInfo[level].Spatials.Length < childCount)
                    _levelInfo[level].Spatials = new SpatialInfo[childCount];

                SpatialInfo[] parentSpatials = _levelInfo[level].Spatials;

                ParallelForSlim(childCount, _parallelPartitions, (start, len) =>
                {
                    var mm = new MinMax(0);

                    for (int b = start; b < len; b++)
                    {
                        var spatial = childLevel.Spatials[childLevel.CellIndex[b]];
                        int idxX = spatial.IdxX >> 1;
                        int idxY = spatial.IdxY >> 1;
                        int morton = MortonNumber(idxX, idxY);

                        mm.Update(idxX, idxY);

                        parentSpatials[b].Set(morton, idxX, idxY, spatial.Index + b);
                    }

                    lock (sync)
                    {
                        minMax.Update(mm);
                    }

                });

                AddGridDims(minMax, level);

                if (_levelInfo[level].CellIndex == null || _levelInfo[level].CellIndex.Length < childCount)
                    _levelInfo[level].CellIndex = new int[childCount + 1000];

                var cellIdx = _levelInfo[level].CellIndex;
                int count = 0;
                int val = int.MaxValue;

                fixed (int* cellIdxPtr = cellIdx)
                fixed (SpatialInfo* spaPtr = parentSpatials)
                {
                    for (int i = 0; i < childCount; i++)
                    {
                        var mort = spaPtr[i].Mort;

                        if (val != mort)
                        {
                            cellIdxPtr[count] = i;
                            val = mort;
                            count++;
                        }
                    }
                }

                cellIdx[count] = childCount;

                _levelInfo[level].CellIndex = cellIdx;
                _levelInfo[level].Spatials = parentSpatials;
                _levelInfo[level].CellCount = count;
            }
        }

        private void WriteCellIndex(LevelInfo[] levelInfo)
        {
            // Writing the cell index as a single large array
            // is much faster than chunking it in at each level.
        
            // Calc total size of cell index.
            long cellIdxLen = 0;

            for (int i = 0; i < levelInfo.Length; i++)
            {
                cellIdxLen += levelInfo[i].CellCount + 1;
            }

            // Build 1D array of cell index.
            if (_allCellIdx.Length < cellIdxLen)
                _allCellIdx = new int[cellIdxLen + 1000]; // Add some padding to reduce future reallocations.

            int cellIdxPos = 0;

            // Append cell index from each level into 1D array.
            for (int i = 0; i < levelInfo.Length; i++)
            {
                var cellIndex = levelInfo[i].CellIndex;
                var count = levelInfo[i].CellCount + 1;

                Array.Copy(cellIndex, 0, _allCellIdx, cellIdxPos, count);

                cellIdxPos += count;
            }

            // Allocate and write to GPU.
            Allocate(ref _gpuCellIdx, cellIdxLen);
            _queue.WriteToBuffer(_allCellIdx, _gpuCellIdx, false, 0, 0, cellIdxLen, null);
        }

        /// <summary>
        /// Populates the first (bottom) level of the mesh on the GPU.
        /// </summary>
        /// <param name="levelInfo">Level info for bottom.</param>
        /// <param name="cellSizeExp">Current cell size exponent.</param>
        /// <remarks>The first level (bottom) cell values are computed from the bodies.</remarks>
        private void BuildBottomLevelGPU(LevelInfo levelInfo, int cellSizeExp)
        {
            int cellCount = levelInfo.CellCount;

            _buildBottomKernel.SetMemoryArgument(0, _gpuInBodies);
            _buildBottomKernel.SetMemoryArgument(1, _gpuOutBodies);
            _buildBottomKernel.SetMemoryArgument(2, _gpuMesh);
            _buildBottomKernel.SetValueArgument(3, cellCount);
            _buildBottomKernel.SetMemoryArgument(4, _gpuCellIdx);
            _buildBottomKernel.SetValueArgument(5, cellSizeExp);

            _queue.Execute(_buildBottomKernel, null, new long[] { BlockCount(cellCount) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
        }

        /// <summary>
        /// Populates the remaining (top) levels of the mesh on the GPU.
        /// </summary>
        /// <param name="levelInfo">Complete collection of level infos.</param>
        /// <param name="cellSizeExp">Current cell size exponent.</param>
        /// <param name="levels">Current number of levels.</param>
        /// <remarks>The top level cell values are computed from the previous level (child) cells.</remarks>
        private void BuildTopLevelsGPU(LevelInfo[] levelInfo, int cellSizeExp, int levels)
        {
            int meshOffset = 0; // Write offset for new cells array location.
            int readOffset = 0; // Read offset for cell and location indexes.

            for (int level = 1; level <= levels; level++)
            {
                int cellSizeExpLevel = cellSizeExp + level;

                meshOffset += levelInfo[level - 1].CellCount;
                _levelIdx[level] = meshOffset;

                int levelOffset = 0;

                levelOffset = _levelIdx[level - 1];
                readOffset += levelInfo[level - 1].CellCount;

                if (level == 1)
                    readOffset += 1;

                LevelInfo current = levelInfo[level];
                int cellCount = current.CellCount;

                _buildTopKernel.SetMemoryArgument(0, _gpuMesh);
                _buildTopKernel.SetValueArgument(1, cellCount);
                _buildTopKernel.SetMemoryArgument(2, _gpuCellIdx);
                _buildTopKernel.SetValueArgument(3, cellSizeExpLevel);
                _buildTopKernel.SetValueArgument(4, levelOffset);
                _buildTopKernel.SetValueArgument(5, meshOffset);
                _buildTopKernel.SetValueArgument(6, readOffset);
                _buildTopKernel.SetValueArgument(7, level);

                _queue.Execute(_buildTopKernel, null, new long[] { BlockCount(cellCount) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
            }
        }

        /// <summary>
        /// Populates a compressed sparse grid array (grid index) and computes the neighbor list for all mesh cells on the GPU.
        /// </summary>
        /// <param name="gridInfo"></param>
        /// <param name="meshSize"></param>
        private void PopGridAndNeighborsGPU(GridInfo[] gridInfo, int meshSize)
        {
            // Calulate total size of 1D mesh neighbor list.
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
            int sizeOfInteger = 4;
            long gridMem = gridSize * sizeOfInteger; // Size of grid index in memory. (n * bytes) (int = 4 bytes)

            // Do we need more than 1 pass?
            if (gridMem > _maxBufferSize)
            {
                passes += (gridMem / _maxBufferSize);
                stride = (int)_gpuGridIndex.Count;
            }

            // Write Grid info to GPU.
            Allocate(ref _gpuGridInfo, gridInfo.Length, true);
            _queue.WriteToBuffer(gridInfo, _gpuGridInfo, false, null);

            int workSize = BlockCount(meshSize) * _threadsPerBlock;

            for (int i = 0; i < passes; i++)
            {
                passOffset = stride * i;

                // Pop compressed grid index array.
                int argi = 0;
                _popGridKernel.SetMemoryArgument(argi++, _gpuGridIndex);
                _popGridKernel.SetValueArgument(argi++, (int)stride);
                _popGridKernel.SetValueArgument(argi++, (int)passOffset);
                _popGridKernel.SetMemoryArgument(argi++, _gpuGridInfo);
                _popGridKernel.SetMemoryArgument(argi++, _gpuMesh);
                _popGridKernel.SetValueArgument(argi++, meshSize);
                _queue.Execute(_popGridKernel, null, new long[] { workSize }, new long[] { _threadsPerBlock }, null);

                // Build neighbor list.
                argi = 0;
                _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuMesh);
                _buildNeighborsKernel.SetValueArgument(argi++, meshSize);
                _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuGridInfo);
                _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuGridIndex);
                _buildNeighborsKernel.SetValueArgument(argi++, (int)stride);
                _buildNeighborsKernel.SetValueArgument(argi++, (int)passOffset);
                _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
                _queue.Execute(_buildNeighborsKernel, null, new long[] { workSize }, new long[] { _threadsPerBlock }, null);

                // We're done with the grid index array, so undo what we added to clear it for the next frame.
                _clearGridKernel.SetMemoryArgument(0, _gpuGridIndex);
                _clearGridKernel.SetValueArgument(1, (int)stride);
                _clearGridKernel.SetValueArgument(2, (int)passOffset);
                _clearGridKernel.SetMemoryArgument(3, _gpuMesh);
                _clearGridKernel.SetValueArgument(4, meshSize);
                _queue.Execute(_clearGridKernel, null, new long[] { workSize }, new long[] { _threadsPerBlock }, null);
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

        private T[] ReadBuffer<T>(ComputeBuffer<T> buffer, bool blocking = false) where T : struct
        {
            T[] buf = new T[buffer.Count];

            _queue.ReadFromBuffer(buffer, ref buf, true, null);
            if (blocking) _queue.Finish();

            return buf;
        }

        private T[] ReadBuffer<T>(ComputeBuffer<T> buffer, long offset, long length) where T : struct
        {
            T[] buf = new T[length - offset];

            _queue.ReadFromBuffer(buffer, ref buf, true, offset, 0, length - offset, null);
            _queue.Finish();

            return buf;
        }

        private ParallelLoopResult ParallelForSlim(int count, int partitions, Action<int, int> body)
        {
            int pLen, pRem, pCount;
            Partition(count, partitions, out pLen, out pRem, out pCount);
            return Parallel.For(0, pCount, (p) =>
            {
                int offset = p * pLen;
                int len = offset + pLen;

                if (p == pCount - 1)
                    len += pRem;

                body(offset, len);
            });
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

        public void Dispose()
        {
            _gpuMesh.Dispose();
            _gpuMeshNeighbors.Dispose();
            _gpuInBodies.Dispose();
            _gpuOutBodies.Dispose();
            _gpuGridIndex.Dispose();
            _gpuCM.Dispose();
            _gpuSortMap.Dispose();
            _gpuCellIdx.Dispose();
            _gpuGridInfo.Dispose();
            _gpuPostNeeded.Dispose();

            _forceKernel.Dispose();
            _collisionSPHKernel.Dispose();
            _collisionElasticKernel.Dispose();
            _popGridKernel.Dispose();
            _buildNeighborsKernel.Dispose();
            _clearGridKernel.Dispose();
            _fixOverlapKernel.Dispose();
            _buildBottomKernel.Dispose();
            _buildTopKernel.Dispose();
            _calcCMKernel.Dispose();
            _reindexKernel.Dispose();

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