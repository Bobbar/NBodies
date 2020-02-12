﻿using Cloo;
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
        private bool _useBrute = false;
        private static int _threadsPerBlock = 256;
        private int _parallelPartitions = 8;//12;
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

        private ComputeKernel _forceKernel;
        private ComputeKernel _collisionSPHKernel;
        private ComputeKernel _collisionElasticKernel;
        private ComputeKernel _popGridKernel;
        private ComputeKernel _clearGridKernel;
        private ComputeKernel _buildNeighborsKernel;
        private ComputeKernel _buildNeighborsBruteKernel;
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
        private ComputeBuffer<Vector3> _gpuCM;
        private ComputeBuffer<int> _gpuSortMap;
        private ComputeBuffer<int> _gpuCellIdx;
        private ComputeBuffer<GridInfo> _gpuGridInfo;
        private ComputeBuffer<int> _gpuPostNeeded;
        private ComputeBuffer<int> _gpuLevelIndex;

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

        public static int GridPasses { get; private set; } = 1;

        public static bool NNUsingBrute { get; private set; } = true;


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
            _buildNeighborsBruteKernel = _program.CreateKernel("BuildNeighborsBrute");
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
            _gpuLevelIndex.Dispose();
            _mesh = new MeshCell[0];
            _bufferInfo.Clear();
            _currentFrame = 0;
            _lastMeshRead = 0;
            InitBuffers();
        }

        private void InitBuffers()
        {
            _gpuGridIndex = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuGridIndex, 0);

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

            _gpuGridInfo = new ComputeBuffer<GridInfo>(_context, ComputeMemoryFlags.ReadOnly, 1);
            Allocate(ref _gpuGridInfo, 0, true);

            _gpuPostNeeded = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuPostNeeded, 1, true);

            _gpuCM = new ComputeBuffer<Vector3>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuCM, 1, true);

            _gpuLevelIndex = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuLevelIndex, 1, true);
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

            // Calc center of mass on GPU from top-most level.
            _calcCMKernel.SetMemoryArgument(0, _gpuMesh);
            _calcCMKernel.SetMemoryArgument(1, _gpuCM);
            _calcCMKernel.SetValueArgument(2, _levelIdx[_levels]);
            _calcCMKernel.SetValueArgument(3, _meshLength);
            _queue.ExecuteTask(_calcCMKernel, null);

            int[] postNeeded = new int[1] { 0 };
            _queue.WriteToBuffer(postNeeded, _gpuPostNeeded, false, null);

            int argi = 0;
            _forceKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _forceKernel.SetValueArgument(argi++, _bodies.Length);
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
            x &= 0x1fffff; // we only look at the first 21 bits
            x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
            x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
            x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
            x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
            x = (x | x << 2) & 0x1249249249249249;

            y &= 0x1fffff;
            y = (y | y << 32) & 0x1f00000000ffff;
            y = (y | y << 16) & 0x1f0000ff0000ff;
            y = (y | y << 8) & 0x100f00f00f00f00f;
            y = (y | y << 4) & 0x10c30c30c30c30c3;
            y = (y | y << 2) & 0x1249249249249249;

            z = (z | z << 32) & 0x1f00000000ffff;
            z = (z | z << 16) & 0x1f0000ff0000ff;
            z = (z | z << 8) & 0x100f00f00f00f00f;
            z = (z | z << 4) & 0x10c30c30c30c30c3;
            z = (z | z << 2) & 0x1249249249249249;

            return x | (y << 1) | (z << 2);
        }

        private void AddGridDims(MinMax minMax, int level)
        {
            int offsetX = (minMax.MinX - 1) * -1;
            int offsetY = (minMax.MinY - 1) * -1;
            int offsetZ = (minMax.MinZ - 1) * -1;

            long columns = Math.Abs(minMax.MinX - minMax.MaxX - 2);
            long rows = Math.Abs(minMax.MinY - minMax.MaxY - 2);
            long layers = Math.Abs(minMax.MinZ - minMax.MaxZ - 2);

            long idxOff = 0;

            if (minMax.MinX < -65000 || minMax.MinY < -65000)
            {
                Debugger.Break();
            }

            if (level == 1)
            {
                idxOff = 0;
            }
            else if (level > 1)
            {
                idxOff = _gridInfo[level - 1].IndexOffset + _gridInfo[level - 1].Size;
            }

            _gridInfo[level].Set(offsetX, offsetY, offsetZ, idxOff, minMax.MinX, minMax.MinY, minMax.MinZ, minMax.MaxX, minMax.MaxY, minMax.MaxZ, columns, rows, layers);
        }

        private void ComputeGridDims(MinMax first, int levels)
        {
            AddGridDims(first, 0);

            for (int level = 1; level <= levels; level++)
            {
                // Reduce dimensions of the previous level to determine the dims of the next level.
                var prev = _gridInfo[level - 1];
                AddGridDims(new MinMax(prev.MinX >> 1, prev.MinY >> 1, prev.MinZ >> 1, prev.MaxX >> 1, prev.MaxY >> 1, prev.MaxZ >> 1), level);
            }
        }

        /// <summary>
        /// Builds the particle mesh and mesh-neighbor index for the current field.
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

            // Compute spatial grid size for top levels.
            long gridSize = 0;
            for (int l = 1; l <= _levels; l++)
            {
                gridSize += _gridInfo[l].Size;
            }

            // Determine if we should use a grid based or brute force nearest neighbor search.
            // Compute a "sparseness" ratio.  Very sparse and large fields will require a large number 
            // of grid passes to complete the neighbor list. In these cases a brute force search may potentially
            // be faster and will require far less memory. 
            long ratio = gridSize / _meshLength;
            long bruteCutoff = 400000;
            if (ratio >= bruteCutoff)
                _useBrute = true;
            else
                _useBrute = false;

            NNUsingBrute = _useBrute;

            // Populate the grid index and mesh neighbor index list.
            if (!_useBrute)
            {
                PopGridAndNeighborsGPU(_gridInfo, _meshLength, gridSize);
            }
            else
            {
                PopNeighborsBrute(_meshLength);
            }
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

            // MinMax object to record field bounds.
            var minMax = new MinMax();
            var sync = new object();

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

                    mm.Update(idxX, idxY, idxZ);

                    _bSpatials[b].Set(morton, idxX, idxY, idxZ, b);
                    _mortKeys[b] = morton;
                }

                lock (sync)
                {
                    minMax.Update(mm);
                }
            });

            // Compute grid dimensions for the rest of the levels and add to grid info array.
            ComputeGridDims(minMax, _levels);

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

            fixed (long* mortPtr = _mortKeys)
            fixed (int* cellIdxPtr = cellIdx)
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
            for (int level = 1; level <= _levels; level++)
            {
                LevelInfo childLevel = _levelInfo[level - 1];
                int childCount = childLevel.CellCount;

                if (_levelInfo[level].Spatials == null || _levelInfo[level].Spatials.Length < childCount)
                    _levelInfo[level].Spatials = new SpatialInfo[childCount];

                SpatialInfo[] parentSpatials = _levelInfo[level].Spatials;

                ParallelForSlim(childCount, _parallelPartitions, (start, len) =>
                {
                    for (int b = start; b < len; b++)
                    {
                        var spatial = childLevel.Spatials[childLevel.CellIndex[b]];
                        int idxX = spatial.IdxX >> 1;
                        int idxY = spatial.IdxY >> 1;
                        int idxZ = spatial.IdxZ >> 1;

                        long morton = MortonNumber(idxX, idxY, idxZ);

                        parentSpatials[b].Set(morton, idxX, idxY, idxZ, spatial.Index + b);
                    }
                });

                if (_levelInfo[level].CellIndex == null || _levelInfo[level].CellIndex.Length < childCount)
                    _levelInfo[level].CellIndex = new int[childCount + 1000];

                var cellIdx = _levelInfo[level].CellIndex;
                int count = 0;
                long val = long.MaxValue;

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

        /// <summary>
        /// Populates a compressed sparse grid array (grid index) and computes the neighbor list for all mesh cells on the GPU.  **USES BRUTE FORCE KERNEL**
        /// </summary>
        /// <param name="meshSize"></param>
        private void PopNeighborsBrute(int meshSize)
        {
            // Calulate total size of 1D mesh neighbor list.
            // Each cell can have a max of 27 neighbors, including itself.
            int topStart = _levelIdx[1];
            int topSize = meshSize - _levelIdx[1];
            int neighborLen = topSize * 27;

            // Reallocate and resize GPU buffer as needed.
            Allocate(ref _gpuMeshNeighbors, neighborLen);

            Allocate(ref _gpuLevelIndex, _levelIdx.Length, true);
            _queue.WriteToBuffer(_levelIdx, _gpuLevelIndex, false, null);

            int workSize = BlockCount(meshSize) * _threadsPerBlock;

            // Build neighbor list.
            int argi = 0;
            _buildNeighborsBruteKernel.SetMemoryArgument(argi++, _gpuMesh);
            _buildNeighborsBruteKernel.SetValueArgument(argi++, meshSize);
            _buildNeighborsBruteKernel.SetMemoryArgument(argi++, _gpuLevelIndex);
            _buildNeighborsBruteKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _buildNeighborsBruteKernel.SetValueArgument(argi++, _levels);
            _buildNeighborsBruteKernel.SetValueArgument(argi++, topStart);
            _queue.Execute(_buildNeighborsBruteKernel, null, new long[] { workSize }, new long[] { _threadsPerBlock }, null);
        }

        /// <summary>
        /// Populates a compressed sparse grid array (grid index) and computes the neighbor list for all mesh cells on the GPU.
        /// </summary>
        /// <param name="gridInfo"></param>
        /// <param name="meshSize"></param>
        private void PopGridAndNeighborsGPU(GridInfo[] gridInfo, int meshSize, long gridSize)
        {
            // Calulate total size of 1D mesh neighbor list.
            // Each cell can have a max of 27 neighbors, including itself.
            int topStart = _levelIdx[1];
            int topSize = meshSize - _levelIdx[1];
            int neighborLen = topSize * 27;

            // Reallocate and resize GPU buffer as needed.
            Allocate(ref _gpuMeshNeighbors, neighborLen);

            // Reallocate and resize GPU buffer as needed.
            long newCap = Allocate(ref _gpuGridIndex, gridSize);
            if (newCap > 0)
            {
                // Fill the new buffer with -1s.
                _queue.FillBuffer(_gpuGridIndex, new int[1] { -1 }, 0, newCap, null);
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

            GridPasses = (int)passes;

            // Write Grid info to GPU.
            Allocate(ref _gpuGridInfo, gridInfo.Length, true);
            _queue.WriteToBuffer(gridInfo, _gpuGridInfo, false, null);

            int workSize = BlockCount(topSize) * _threadsPerBlock;

            for (int i = 0; i < passes; i++)
            {
                passOffset = stride * i;

                // Pop compressed grid index array.
                int argi = 0;
                _popGridKernel.SetMemoryArgument(argi++, _gpuGridIndex);
                _popGridKernel.SetValueArgument(argi++, stride);
                _popGridKernel.SetValueArgument(argi++, passOffset);
                _popGridKernel.SetMemoryArgument(argi++, _gpuGridInfo);
                _popGridKernel.SetMemoryArgument(argi++, _gpuMesh);
                _popGridKernel.SetValueArgument(argi++, meshSize);
                _popGridKernel.SetValueArgument(argi++, topStart);
                _queue.Execute(_popGridKernel, null, new long[] { workSize }, new long[] { _threadsPerBlock }, null);

                // Build neighbor list.
                argi = 0;
                _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuMesh);
                _buildNeighborsKernel.SetValueArgument(argi++, meshSize);
                _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuGridInfo);
                _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuGridIndex);
                _buildNeighborsKernel.SetValueArgument(argi++, stride);
                _buildNeighborsKernel.SetValueArgument(argi++, passOffset);
                _buildNeighborsKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
                _buildNeighborsKernel.SetValueArgument(argi++, topStart);
                _queue.Execute(_buildNeighborsKernel, null, new long[] { workSize }, new long[] { _threadsPerBlock }, null);

                // We're done with the grid index array, so undo what we added to clear it for the next pass.
                _clearGridKernel.SetMemoryArgument(0, _gpuGridIndex);
                _clearGridKernel.SetValueArgument(1, stride);
                _clearGridKernel.SetValueArgument(2, passOffset);
                _clearGridKernel.SetMemoryArgument(3, _gpuMesh);
                _clearGridKernel.SetValueArgument(4, meshSize);
                _clearGridKernel.SetValueArgument(5, topStart);
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
            _gpuGridIndex.Dispose();
            _gpuCM.Dispose();
            _gpuSortMap.Dispose();
            _gpuCellIdx.Dispose();
            _gpuGridInfo.Dispose();
            _gpuPostNeeded.Dispose();
            _gpuLevelIndex.Dispose();

            _forceKernel.Dispose();
            _collisionSPHKernel.Dispose();
            _collisionElasticKernel.Dispose();
            _popGridKernel.Dispose();
            _buildNeighborsKernel.Dispose();
            _buildNeighborsBruteKernel.Dispose();
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