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
        private bool _useFastMath = true;
        private int _gpuIndex = 4;
        private int _levels = 4;
        private static int _threadsPerBlock = 256;
        private int _parallelPartitions = 14;//12;
        private long _maxBufferSize = 0;
        private float _kernelSize = 1.0f;
        private SPHPreCalc _preCalcs;

        private int[] _levelIdx = new int[0]; // Locations of each level within the 1D mesh array.
        private MeshCell[] _mesh = new MeshCell[0]; // 1D array of mesh cells. (Populated on GPU, and read for UI display only.)
        private int _meshLength = 0; // Total length of the 1D mesh array.
        private GridInfo[] _gridInfo = new GridInfo[0]; // Bounds and offsets for each level of the neighbor list grid.
        private LevelInfo[] _levelInfo = new LevelInfo[0]; // Spatial info, cell index map and counts for each level.
        private Body[] _bodies = new Body[0]; // Local reference for the current body array.

        // Fields for large allocations.
        private SpatialInfo[] _bSpatials = new SpatialInfo[0]; // Body spatials.
        private long[] _mortKeys = new long[0]; // Array of Morton numbers for spatials sorting.
        private int[] _allCellIdx = new int[0]; // Buffer for completed/flattened cell index map.

        private ManualResetEventSlim _meshReadyWait = new ManualResetEventSlim(false);

        private ComputeContext _context;
        private ComputeCommandQueue _queue;
        private ComputeDevice _device = null;
        private ComputeProgram _program;

        private ComputeKernel _forceKernelLocal;
        private ComputeKernel _forceKernelFar;
        private ComputeKernel _collisionSPHKernel;
        private ComputeKernel _collisionElasticKernel;
        private ComputeKernel _buildNeighborsMeshKernel;
        private ComputeKernel _fixOverlapKernel;
        private ComputeKernel _buildBottomKernel;
        private ComputeKernel _buildTopKernel;
        private ComputeKernel _calcCMKernel;
        private ComputeKernel _reindexKernel;

        private ComputeBuffer<MeshCell> _gpuMesh;
        private ComputeBuffer<int> _gpuMeshNeighbors;
        private ComputeBuffer<Body> _gpuInBodies;
        private ComputeBuffer<Body> _gpuOutBodies;
        private ComputeBuffer<Vector3> _gpuCM;
        private ComputeBuffer<int> _gpuSortMap;
        private ComputeBuffer<int> _gpuCellIdx;
        private ComputeBuffer<int> _gpuPostNeeded;

        private static Dictionary<long, BufferDims> _bufferInfo = new Dictionary<long, BufferDims>();

        private Stopwatch timer = new Stopwatch();
        private Stopwatch timer2 = new Stopwatch();

        private long _currentFrame = 0;
        private long _lastMeshRead = 0;

        public MeshCell[] CurrentMesh
        {
            get
            {
                // Only read mesh from GPU if it has changed.
                if (_lastMeshRead != _currentFrame)
                {
                    // Read it now if it's ready.
                    if (_meshReadyWait.IsSet)
                    {
                        ReadMesh();
                    }
                    else
                    {
                        // Wait until it's ready then read.
                        _meshReadyWait.Wait();
                        ReadMesh();
                    }

                    _lastMeshRead = _currentFrame;
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

        public OpenCLPhysics(ComputeDevice device, int threadsperblock, bool useFastMath)
        {
            _device = device;
            _useFastMath = useFastMath;

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
                string options;

                if (_useFastMath)
                    options = "-cl-std=CL1.2 -cl-fast-relaxed-math -D FASTMATH";
                else
                    options = "-cl-std=CL1.2";

                _program.Build(null, options, null, IntPtr.Zero);
            }
            catch (BuildProgramFailureComputeException ex)
            {
                string buildLog = _program.GetBuildLog(_device);
                System.IO.File.WriteAllText("build_error.txt", buildLog);
                Console.WriteLine(buildLog);
                throw;
            }

            //var bins = Encoding.UTF8.GetString(_program.Binaries[0]);
            //File.WriteAllText(Environment.CurrentDirectory + "/Physics/Kernels.ptx", bins);

            Console.WriteLine(_program.GetBuildLog(_device));
            System.IO.File.WriteAllText("build_log.txt", _program.GetBuildLog(_device));

            _forceKernelLocal = _program.CreateKernel("CalcForceLocal");
            _forceKernelFar = _program.CreateKernel("CalcForceFar");
            _collisionSPHKernel = _program.CreateKernel("SPHCollisions");
            _collisionElasticKernel = _program.CreateKernel("ElasticCollisions");
            _buildNeighborsMeshKernel = _program.CreateKernel("BuildNeighborsMesh");
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
            _gpuCM.Dispose();
            _gpuSortMap.Dispose();
            _gpuCellIdx.Dispose();
            _gpuPostNeeded.Dispose();
            _mesh = new MeshCell[0];
            _bufferInfo.Clear();
            _currentFrame = 0;
            _lastMeshRead = 0;
            InitBuffers();
        }

        private void InitBuffers()
        {
            _gpuMeshNeighbors = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMeshNeighbors, 0);

            _gpuInBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuInBodies, 0, true);

            _gpuOutBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuOutBodies, 0, true);

            _gpuMesh = new ComputeBuffer<MeshCell>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMesh, 0, true);

            _gpuSortMap = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadOnly, 1);
            Allocate(ref _gpuSortMap, 0, true);

            _gpuCellIdx = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadOnly, 1);
            Allocate(ref _gpuCellIdx, 0, true);

            _gpuPostNeeded = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuPostNeeded, 1, true);

            _gpuCM = new ComputeBuffer<Vector3>(_context, ComputeMemoryFlags.ReadWrite, 1);
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

            _currentFrame++;


            int[] postNeeded = new int[1] { 0 };
            _queue.WriteToBuffer(postNeeded, _gpuPostNeeded, false, null);

            // Compute gravity and SPH forces for the near/local field.
            int argi = 0;
            _forceKernelLocal.SetMemoryArgument(argi++, _gpuInBodies);
            _forceKernelLocal.SetValueArgument(argi++, _bodies.Length);
            _forceKernelLocal.SetMemoryArgument(argi++, _gpuMesh);
            _forceKernelLocal.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _forceKernelLocal.SetValueArgument(argi++, sim);
            _forceKernelLocal.SetValueArgument(argi++, _preCalcs);
            _queue.Execute(_forceKernelLocal, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);

            // Compute gravity forces for the far/distant field.
            argi = 0;
            int meshTopStart = _levelIdx[_levels];
            int meshTopEnd = _meshLength;
            _forceKernelFar.SetMemoryArgument(argi++, _gpuInBodies);
            _forceKernelFar.SetValueArgument(argi++, _bodies.Length);
            _forceKernelFar.SetMemoryArgument(argi++, _gpuMesh);
            _forceKernelFar.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _forceKernelFar.SetValueArgument(argi++, meshTopStart);
            _forceKernelFar.SetValueArgument(argi++, meshTopEnd);
            _forceKernelFar.SetMemoryArgument(argi++, _gpuPostNeeded);
            _queue.Execute(_forceKernelFar, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);

            // Compute elastic collisions.
            argi = 0;
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _collisionElasticKernel.SetValueArgument(argi++, _bodies.Length);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMesh);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _collisionElasticKernel.SetValueArgument(argi++, Convert.ToInt32(sim.CollisionsOn));
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuPostNeeded);
            _queue.Execute(_collisionElasticKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);

            // Compute SPH forces/collisions.
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
            using (var inBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, bodies.Length))
            using (var outBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, bodies.Length))
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

        private void ReadMesh()
        {
            if (_mesh.Length != _meshLength)
                _mesh = new MeshCell[_meshLength];

            _queue.ReadFromBuffer(_gpuMesh, ref _mesh, false, 0, 0, _meshLength, null);
            _queue.Finish();
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private long MortonNumber(long x, long y, long z)
        {
            x &= 0x1fffff;
            x = (x | x << 32) & 0x1f00000000ffff;
            x = (x | x << 16) & 0x1f0000ff0000ff;
            x = (x | x << 8) & 0x100f00f00f00f00f;
            x = (x | x << 4) & 0x10c30c30c30c30c3;
            x = (x | x << 2) & 0x1249249249249249;

            y &= 0x1fffff;
            y = (y | y << 32) & 0x1f00000000ffff;
            y = (y | y << 16) & 0x1f0000ff0000ff;
            y = (y | y << 8) & 0x100f00f00f00f00f;
            y = (y | y << 4) & 0x10c30c30c30c30c3;
            y = (y | y << 2) & 0x1249249249249249;

            z &= 0x1fffff;
            z = (z | z << 32) & 0x1f00000000ffff;
            z = (z | z << 16) & 0x1f0000ff0000ff;
            z = (z | z << 8) & 0x100f00f00f00f00f;
            z = (z | z << 4) & 0x10c30c30c30c30c3;
            z = (z | z << 2) & 0x1249249249249249;

            return x | (y << 1) | (z << 2);
        }

        /// <summary>
        /// Builds the particle mesh and mesh-neighbor index for the current field.
        /// </summary>
        /// <param name="cellSizeExp">Cell size exponent. 2 ^ exponent = cell size.</param>
        private void BuildMesh(int cellSizeExp)
        {
            // Mesh info for each level.
            if (_levelInfo.Length != (_levels + 1))
                _levelInfo = new LevelInfo[_levels + 1];

            // [Bottom level]
            // Compute cell count, cell index map and sort map.
            ComputeBodySpatials(cellSizeExp);

            // Start reindexing bodies on the GPU.
            ReindexBodiesGPU();

            // [Upper levels]
            // Compute cell count and cell index maps for all upper levels.
            ComputeUpperSpatials();

            // Get the total number of mesh cells to be created.
            int totCells = 0;
            foreach (var lvl in _levelInfo)
                totCells += lvl.CellCount;

            _meshLength = totCells;

            // Allocate the mesh buffer on the GPU.
            Allocate(ref _gpuMesh, _meshLength, false);

            // Index to hold the starting indexes for each level within the 1D mesh array.
            _levelIdx = new int[_levels + 1];
            _levelIdx[0] = 0;

            // Write the cell index map as one large array to the GPU.
            WriteCellIndex(_levelInfo);

            // Build the first (bottom) level of the mesh.
            BuildBottomLevelGPU(_levelInfo[0], cellSizeExp);

            // Build the remaining (upper) levels of the mesh.
            BuildUpperLevelsGPU(_levelInfo, cellSizeExp, _levels);

            // Calc center of mass on GPU from top-most level.
            _calcCMKernel.SetMemoryArgument(0, _gpuMesh);
            _calcCMKernel.SetMemoryArgument(1, _gpuCM);
            _calcCMKernel.SetValueArgument(2, _levelIdx[_levels]);
            _calcCMKernel.SetValueArgument(3, _meshLength);
            _queue.ExecuteTask(_calcCMKernel, null);

            // Build Nearest Neighbor List.
            PopNeighborsMeshGPU(_meshLength);
        }

        /// <summary>
        /// Computes spatial info (Morton number, X/Y indexes, mesh cell map and sort map) for all bodies.
        /// </summary>
        /// <param name="cellSizeExp">Cell size exponent. "Math.Pow(2, exponent)"</param>
        private unsafe void ComputeBodySpatials(int cellSizeExp)
        {
            // Spatial info to be computed.
            if (_bSpatials.Length < _bodies.Length)
                _bSpatials = new SpatialInfo[_bodies.Length];

            // Array of morton numbers used for sorting.
            // Using a key array when sorting is much faster than sorting an array of objects by a field.
            if (_mortKeys.Length < _bodies.Length)
                _mortKeys = new long[_bodies.Length];

            // Compute the spatial info in parallel.
            ParallelForSlim(_bodies.Length, _parallelPartitions, (start, len) =>
            {
                var mm = new MinMax();

                for (int b = start; b < len; b++)
                {
                    int idxX = (int)Math.Floor(_bodies[b].PosX) >> cellSizeExp;
                    int idxY = (int)Math.Floor(_bodies[b].PosY) >> cellSizeExp;
                    int idxZ = (int)Math.Floor(_bodies[b].PosZ) >> cellSizeExp;

                    long morton = MortonNumber(idxX, idxY, idxZ);

                    _bSpatials[b].Set(morton, idxX, idxY, idxZ, b);
                    _mortKeys[b] = morton;
                }
            });

            // Sort by morton number to produce a spatially sorted array.
            Sort.ParallelQuickSort(_mortKeys, _bSpatials, _bodies.Length);

            // Compute number of unique morton numbers to determine cell count,
            // and build the start index of each cell.
            int count = 0;
            long val = long.MaxValue;

            // Allocate then map the sort map buffer so we can write to it directly.
            Allocate(ref _gpuSortMap, _bodies.Length + 100);
            var sortMapPtr = _queue.Map(_gpuSortMap, true, ComputeMemoryMappingFlags.Write, 0, _gpuSortMap.Count, null);
            var sortMapNativePtr = (int*)sortMapPtr.ToPointer();

            // Check cellindex allocation.
            var cellIdx = _levelInfo[0].CellIndex;
            if (cellIdx == null || cellIdx.Length < _bodies.Length)
                cellIdx = new int[_bodies.Length + 100];

            // This loop is a bit faster if we pin the arrays.
            fixed (long* mortPtr = _mortKeys)
            fixed (int* cellIdxPtr = cellIdx)
            fixed (SpatialInfo* spaPtr = _bSpatials)
            {
                for (int i = 0; i < _bodies.Length; i++)
                {
                    var mort = mortPtr[i];
                    sortMapNativePtr[i] = spaPtr[i].Index;

                    // Find the start of each new morton number and record location to build the cell index map.
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

            // Set the computed info.
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
        private unsafe void ComputeUpperSpatials()
        {
            // Iterate and compute spatial info and cell index map for upper (parent) mesh levels.
            for (int level = 1; level <= _levels; level++)
            {
                // Get child level spatials.
                LevelInfo childLevel = _levelInfo[level - 1];
                int childCount = childLevel.CellCount;

                // Allocate spatials buffer for the parent level about to be computed.
                if (_levelInfo[level].Spatials == null || _levelInfo[level].Spatials.Length < childCount)
                    _levelInfo[level].Spatials = new SpatialInfo[childCount];

                // Grab a pointer for the spatials.
                SpatialInfo[] parentSpatials = _levelInfo[level].Spatials;

                // Compute parent spatials in parallel.
                // We don't need to read every element of the child spatials.
                // Use the cell index map and count computed for the previous (child) level. 
                ParallelForSlim(childCount, _parallelPartitions, (start, len) =>
                {
                    for (int p = start; p < len; p++)
                    {
                        // Read child level spatials at each cell location
                        // and compute parent spatials.
                        var spatial = childLevel.Spatials[childLevel.CellIndex[p]];
                        int idxX = spatial.IdxX >> 1;
                        int idxY = spatial.IdxY >> 1;
                        int idxZ = spatial.IdxZ >> 1;

                        long morton = MortonNumber(idxX, idxY, idxZ);

                        parentSpatials[p].Set(morton, idxX, idxY, idxZ, spatial.Index + p);
                    }
                });

                // Allocate cell index map buffer for the parent level.
                if (_levelInfo[level].CellIndex == null || _levelInfo[level].CellIndex.Length < childCount)
                    _levelInfo[level].CellIndex = new int[childCount + 1000];

                var cellIdx = _levelInfo[level].CellIndex;
                int count = 0;
                long val = long.MaxValue;

                // Compute cell index map and count for the parent.
                // Pin for speed.
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

                // Add the last cell index value;
                cellIdx[count] = childCount;

                // Write info for this level.
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
                cellIdxLen += levelInfo[i].CellCount + 1;

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
            _buildBottomKernel.SetMemoryArgument(1, _gpuMesh);
            _buildBottomKernel.SetValueArgument(2, cellCount);
            _buildBottomKernel.SetMemoryArgument(3, _gpuCellIdx);
            _buildBottomKernel.SetValueArgument(4, cellSizeExp);
            _buildBottomKernel.SetValueArgument(5, (int)Math.Pow(2.0f, cellSizeExp));

            _queue.Execute(_buildBottomKernel, null, new long[] { BlockCount(cellCount) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
        }

        /// <summary>
        /// Populates the remaining (top) levels of the mesh on the GPU.
        /// </summary>
        /// <param name="levelInfo">Complete collection of level infos.</param>
        /// <param name="cellSizeExp">Current cell size exponent.</param>
        /// <param name="levels">Current number of levels.</param>
        /// <remarks>The top level cell values are computed from the previous level (child) cells.</remarks>
        private void BuildUpperLevelsGPU(LevelInfo[] levelInfo, int cellSizeExp, int levels)
        {
            int cellIdxOffset = 0; // Read offset for cell index map.
            int levelOffset = 0; // Offset for previous level mesh locations.  ( cellIndex[n] + levelOffset = mesh cells @ level - 1 )

            for (int level = 1; level <= levels; level++)
            {
                int cellSizeExpLevel = cellSizeExp + level;
                LevelInfo parentLevel = levelInfo[level];
                LevelInfo childLevel = levelInfo[level - 1];

                // Collect mesh level start indices.
                _levelIdx[level] = _levelIdx[level - 1] + childLevel.CellCount;

                // Get offsets for this level.
                cellIdxOffset += childLevel.CellCount + 1;
                levelOffset = _levelIdx[level - 1];

                _buildTopKernel.SetMemoryArgument(0, _gpuMesh);
                _buildTopKernel.SetValueArgument(1, parentLevel.CellCount);
                _buildTopKernel.SetMemoryArgument(2, _gpuCellIdx);
                _buildTopKernel.SetValueArgument(3, cellSizeExpLevel);
                _buildTopKernel.SetValueArgument(4, (int)Math.Pow(2.0f, cellSizeExpLevel));
                _buildTopKernel.SetValueArgument(5, levelOffset);
                _buildTopKernel.SetValueArgument(6, cellIdxOffset);
                _buildTopKernel.SetValueArgument(7, level);

                _queue.Execute(_buildTopKernel, null, new long[] { BlockCount(parentLevel.CellCount) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
            }
        }


        private void PopNeighborsMeshGPU(int meshSize)
        {
            // Calulate total size of 1D mesh neighbor list.
            // Each cell can have a max of 27 neighbors, including itself.
            int topSize = meshSize - _levelIdx[1];
            int neighborLen = topSize * 27;

            // Reallocate and resize GPU buffer as needed.
            Allocate(ref _gpuMeshNeighbors, neighborLen);

            // Start at the top level and move down.
            for (int level = _levels; level >= 1; level--)
            {
                // Compute the bounds and size of this invocation.
                int start = _levelIdx[level];
                int end = meshSize;

                if (level < _levels)
                    end = _levelIdx[level + 1];

                int cellCount = end - start;
                int workSize = BlockCount(cellCount) * _threadsPerBlock;

                // Populate the neighbor list for this level.
                int argi = 0;
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMesh);
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, _levelIdx[1]);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, _levels);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, level);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, start);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, end);
                _queue.Execute(_buildNeighborsMeshKernel, null, new long[] { workSize }, new long[] { _threadsPerBlock }, null);
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

                    buffer = new ComputeBuffer<T>(_context, flags, newCapacity);

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

                    buffer = new ComputeBuffer<T>(_context, flags, size);

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
            _gpuCM.Dispose();
            _gpuSortMap.Dispose();
            _gpuCellIdx.Dispose();
            _gpuPostNeeded.Dispose();

            _forceKernelLocal.Dispose();
            _forceKernelFar.Dispose();
            _collisionSPHKernel.Dispose();
            _collisionElasticKernel.Dispose();
            _buildNeighborsMeshKernel.Dispose();
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