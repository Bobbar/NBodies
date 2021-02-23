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
using System.Linq;

namespace NBodies.Physics
{
    public class OpenCLPhysics : IPhysicsCalc, IDisposable
    {
        private bool _useFastMath = true;
        private int _gpuIndex = 4;
        private int _levels = 4;
        private static int _threadsPerBlock = 256;
        private long _maxBufferSize = 0;
        private float _kernelSize = 1.0f;
        private SPHPreCalc _preCalcs;
        private const int SIZEOFINT = 4;
        private const float BUF_GROW_FACTOR = 1.4f;

        private int[] _levelIdx = new int[0]; // Locations of each level within the 1D mesh array.
        private MeshCell[] _mesh = new MeshCell[0]; // 1D array of mesh cells. (Populated on GPU, and read for UI display only.)
        private int _meshLength = 0; // Total length of the 1D mesh array.
        private Body[] _bodies = new Body[0]; // Local reference for the current body array.

        private ManualResetEventSlim _meshReadyWait = new ManualResetEventSlim(false);

        private ComputeContext _context;
        private ComputeCommandQueue _queue;
        private ComputeDevice _device = null;
        private ComputeProgram _program;

        private ComputeKernel _forceKernel;
        private ComputeKernel _collisionSPHKernel;
        private ComputeKernel _collisionElasticKernel;
        private ComputeKernel _buildNeighborsMeshKernel;
        private ComputeKernel _fixOverlapKernel;
        private ComputeKernel _buildBottomKernel;
        private ComputeKernel _buildTopKernel;
        private ComputeKernel _calcCMKernel;
        private ComputeKernel _reindexKernel;
        private ComputeKernel _cellMapKernel;
        private ComputeKernel _compressCellMapKernel;
        private ComputeKernel _computeMortsKernel;

        private ComputeBuffer<MeshCell> _gpuMesh;
        private ComputeBuffer<int> _gpuMeshNeighbors;
        private ComputeBuffer<Body> _gpuInBodies;
        private ComputeBuffer<Body> _gpuOutBodies;
        private ComputeBuffer<Vector3> _gpuCM;
        private ComputeBuffer<int> _gpuPostNeeded;
        private ComputeBuffer<int> _gpuMap;
        private ComputeBuffer<int> _gpuMapFlat;
        private ComputeBuffer<int> _gpuCounts;
        private ComputeBuffer<long> _gpuParentMorts;
        private ComputeBuffer<long2> _gpuBodyMorts;

        private Stopwatch timer = new Stopwatch();
        private Stopwatch timer2 = new Stopwatch();

        private long _currentFrame = 0;
        private long _lastMeshRead = 0;

        private Dictionary<int, ComputeKernel> _sortKerns = new Dictionary<int, ComputeKernel>();

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
                string options;

                if (_useFastMath)
                    options = $@"-cl-std=CL1.2 -cl-fast-relaxed-math -D FASTMATH";
                else
                    options = $"-cl-std=CL1.2";

                _program.Build(new[] { _device }, options, null, IntPtr.Zero);
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
            _buildNeighborsMeshKernel = _program.CreateKernel("BuildNeighborsMesh");
            _fixOverlapKernel = _program.CreateKernel("FixOverlaps");
            _buildBottomKernel = _program.CreateKernel("BuildBottom");
            _buildTopKernel = _program.CreateKernel("BuildTop");
            _calcCMKernel = _program.CreateKernel("CalcCenterOfMass");
            _reindexKernel = _program.CreateKernel("ReindexBodies");
            _cellMapKernel = _program.CreateKernel("MapMorts");
            _compressCellMapKernel = _program.CreateKernel("CompressMap");
            _computeMortsKernel = _program.CreateKernel("ComputeMorts");

            _sortKerns.Add(12, _program.CreateKernel("ParallelBitonic_C4"));
            _sortKerns.Add(10, _program.CreateKernel("ParallelBitonic_B16"));
            _sortKerns.Add(9, _program.CreateKernel("ParallelBitonic_B8"));
            _sortKerns.Add(8, _program.CreateKernel("ParallelBitonic_B4"));
            _sortKerns.Add(7, _program.CreateKernel("ParallelBitonic_B2"));

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
            _gpuPostNeeded.Dispose();

            _gpuCounts.Dispose();
            _gpuParentMorts.Dispose();
            _gpuMap.Dispose();
            _gpuMapFlat.Dispose();
            _gpuBodyMorts.Dispose();

            _mesh = new MeshCell[0];
            _currentFrame = 0;
            _lastMeshRead = 0;
            InitBuffers();
        }

        public void Dispose()
        {
            _gpuMesh.Dispose();
            _gpuMeshNeighbors.Dispose();
            _gpuInBodies.Dispose();
            _gpuOutBodies.Dispose();
            _gpuCM.Dispose();
            _gpuPostNeeded.Dispose();
            _gpuCounts.Dispose();
            _gpuParentMorts.Dispose();
            _gpuMap.Dispose();
            _gpuMapFlat.Dispose();
            _gpuBodyMorts.Dispose();


            _forceKernel.Dispose();
            _collisionSPHKernel.Dispose();
            _collisionElasticKernel.Dispose();
            _buildNeighborsMeshKernel.Dispose();
            _fixOverlapKernel.Dispose();
            _buildBottomKernel.Dispose();
            _buildTopKernel.Dispose();
            _calcCMKernel.Dispose();
            _reindexKernel.Dispose();
            _cellMapKernel.Dispose();
            _compressCellMapKernel.Dispose();
            _computeMortsKernel.Dispose();

            foreach (var kern in _sortKerns)
                kern.Value.Dispose();

            _program.Dispose();
            _context.Dispose();
            _queue.Dispose();
        }


        private void InitBuffers()
        {
            _gpuMeshNeighbors = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMeshNeighbors, 1);

            _gpuInBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuInBodies, 1, true);

            _gpuOutBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuOutBodies, 1, true);

            _gpuMesh = new ComputeBuffer<MeshCell>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMesh, 1, true);

            _gpuPostNeeded = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuPostNeeded, 1, true);

            _gpuCM = new ComputeBuffer<Vector3>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuCM, 1, true);

            _gpuBodyMorts = new ComputeBuffer<long2>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuBodyMorts, 1, true);

            _gpuMap = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMap, 1, true);

            _gpuMapFlat = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMapFlat, 1, true);

            _gpuCounts = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuCounts, 1, true);

            _gpuParentMorts = new ComputeBuffer<long>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuParentMorts, 1, true);
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

            // Post process flag.
            // Set by kernels when host side post processing is needed. (Culled and/or fractured bodies present)
            int[] postNeeded = new int[1] { 0 };
            _queue.WriteToBuffer(postNeeded, _gpuPostNeeded, false, null);

            // Get start and end of top level mesh cells.
            int meshTopStart = _levelIdx[_levels];
            int meshTopEnd = _meshLength;

            // Compute gravity and SPH forces for the near/local field.
            int argi = 0;
            _forceKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _forceKernel.SetValueArgument(argi++, _bodies.Length);
            _forceKernel.SetMemoryArgument(argi++, _gpuMesh);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _forceKernel.SetValueArgument(argi++, sim);
            _forceKernel.SetValueArgument(argi++, _preCalcs);
            _forceKernel.SetValueArgument(argi++, meshTopStart);
            _forceKernel.SetValueArgument(argi++, meshTopEnd);
            _forceKernel.SetMemoryArgument(argi++, _gpuPostNeeded);
            _queue.Execute(_forceKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);

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

        /// <summary>
        /// Builds the particle mesh and mesh-neighbor index for the current field.
        /// </summary>
        /// <param name="cellSizeExp">Cell size exponent. 2 ^ exponent = cell size.</param>
        private void BuildMesh(int cellSizeExp)
        {
            // Index of mesh level locations.
            _levelIdx = new int[_levels + 1];
            _levelIdx[0] = 0; // Bottom level.

            // Compute a padded size.
            // The current sort kernel has particular requirements for the input size
            int padLen = ComputePaddedSize(_bodies.Length);

            // Compute Z-Order morton numbers for bodies.
            ComputeMortsGPU(padLen, cellSizeExp);

            // Sort by the morton numbers.
            SortByMortGPU(padLen);
            ReindexBodiesGPU();

            // Build each level of the mesh.
            BuildMeshGPU(cellSizeExp);

            // Calc center of mass on GPU from top-most level.
            _calcCMKernel.SetMemoryArgument(0, _gpuMesh);
            _calcCMKernel.SetMemoryArgument(1, _gpuCM);
            _calcCMKernel.SetValueArgument(2, _levelIdx[_levels]);
            _calcCMKernel.SetValueArgument(3, _meshLength);
            _queue.ExecuteTask(_calcCMKernel, null);

            // Build Nearest Neighbor List.
            PopNeighborsMeshGPU(_meshLength);
        }

        private int ComputePaddedSize(int len)
        {
            const int maxLen = 256 << 14;

            for (int n = 64; n < maxLen; n <<= 1)
            {
                if (len <= n)
                    return n;
            }

            return maxLen;
        }

        private void ComputeMortsGPU(int padLen, int cellSizeExp)
        {
            Allocate(ref _gpuBodyMorts, padLen, exact: true);

            _computeMortsKernel.SetMemoryArgument(0, _gpuOutBodies);
            _computeMortsKernel.SetValueArgument(1, _bodies.Length);
            _computeMortsKernel.SetValueArgument(2, padLen);
            _computeMortsKernel.SetValueArgument(3, cellSizeExp);
            _computeMortsKernel.SetMemoryArgument(4, _gpuBodyMorts);
            _queue.Execute(_computeMortsKernel, null, new long[] { BlockCount(padLen) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
        }
        private void SortByMortGPU(int padLen)
        {
            //
            // Credit & Thanks to:
            // Eric Bainville - OpenCL Sorting
            // http://www.bealto.com/gpu-sorting_intro.html
            //

            // Allowed sort kernel bit mask.
            // B2 + B4 + B8 + B16 = 30
            // Note: B16 may not be compatible with all hardware.
            const int ALLOWB = 30; //14;

            int sz = Marshal.SizeOf<long2>();
            int n = padLen;
            for (int length = 1; length < n; length <<= 1)
            {
                int inc = length;
                var strategy = new List<int>();

                int ii = inc;
                while (ii > 0)
                {

                    if (ii == 128 || ii == 32 || ii == 8)
                    {
                        strategy.Add(-1);
                        break;
                    }

                    int d = 1;
                    // default is 1 bit
                    if (false) d = 1;

                    // Force jump to 128
                    else if (ii == 256) d = 1;
                    else if (ii == 512 && Convert.ToBoolean(ALLOWB & 4)) d = 2;
                    else if (ii == 1024 && Convert.ToBoolean(ALLOWB & 8)) d = 3;
                    else if (ii == 2048 && Convert.ToBoolean(ALLOWB & 16)) d = 4;

                    else if (ii >= 8 && Convert.ToBoolean(ALLOWB & 16)) d = 4;
                    else if (ii >= 4 && Convert.ToBoolean(ALLOWB & 8)) d = 3;
                    else if (ii >= 2 && Convert.ToBoolean(ALLOWB & 4)) d = 2;
                    else d = 1;
                    strategy.Add(d);
                    ii >>= d;
                }

                while (inc > 0)
                {
                    int ninc = 0;
                    ComputeKernel kid = _sortKerns[12];
                    int doLocal = 0;
                    int nThreads = 0;
                    int d = strategy.First(); strategy.RemoveAt(0);

                    switch (d)
                    {
                        case -1:
                            kid = _sortKerns[12];
                            ninc = -1; // reduce all bits
                            doLocal = 4;
                            nThreads = n >> 2;
                            break;

                        case 4:
                            kid = _sortKerns[10];
                            ninc = 4;
                            nThreads = n >> ninc;
                            break;

                        case 3:
                            kid = _sortKerns[9];
                            ninc = 3;
                            nThreads = n >> ninc;
                            break;

                        case 2:
                            kid = _sortKerns[8];
                            ninc = 2;
                            nThreads = n >> ninc;
                            break;

                        case 1:
                            kid = _sortKerns[7];
                            ninc = 1;
                            nThreads = n >> ninc;
                            break;

                        default:
                            Debugger.Break();
                            break;
                    }
                    int wg = (int)_device.MaxWorkGroupSize;
                    wg = Math.Min(wg, 256);
                    wg = Math.Min(wg, nThreads);
                    kid.SetMemoryArgument(0, _gpuBodyMorts);
                    kid.SetValueArgument(1, inc);
                    kid.SetValueArgument(2, length << 1);

                    if (doLocal > 0)
                        kid.SetLocalArgument(3, doLocal * wg * sz);

                    _queue.Execute(kid, null, new long[] { nThreads, 1 }, new long[] { wg, 1 }, null);

                    if (ninc < 0) break; // done
                    inc >>= ninc;
                }
            }
        }

        private void BuildMeshGPU(int cellSizeExp)
        {
            // First level map is built from body morts.
            // Subsequent parent level maps are built from child cell morts.

            // Bottom level block count.
            int blocks = BlockCount(_bodies.Length);

            // Allocate map and count buffers.
            Allocate(ref _gpuMap, _bodies.Length, false);
            Allocate(ref _gpuMapFlat, _bodies.Length, false);
            Allocate(ref _gpuCounts, blocks, false);

            // Build initial map from sorted body morts.
            _cellMapKernel.SetMemoryArgument(0, _gpuBodyMorts);
            _cellMapKernel.SetValueArgument(1, _bodies.Length);
            _cellMapKernel.SetMemoryArgument(2, _gpuMap);
            _cellMapKernel.SetMemoryArgument(3, _gpuCounts);
            _cellMapKernel.SetLocalArgument(4, SIZEOFINT * _threadsPerBlock);
            _cellMapKernel.SetValueArgument(5, 2); // Set step size to 2 for long2 input type.
            _queue.Execute(_cellMapKernel, null, new long[] { blocks * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);

            // Remove the gaps to compress the cell map into the beginning of the buffer.
            // This allows the map to be read properly by the mesh building kernels.
            _compressCellMapKernel.SetValueArgument(0, blocks);
            _compressCellMapKernel.SetMemoryArgument(1, _gpuMap);
            _compressCellMapKernel.SetMemoryArgument(2, _gpuMapFlat);
            _compressCellMapKernel.SetMemoryArgument(3, _gpuCounts);
            _queue.Execute(_compressCellMapKernel, null, new long[] { BlockCount(blocks) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);

            // Read the counts computed by each block and compute the total count.
            var counts = ReadBuffer(_gpuCounts, 0, blocks, false);  // Should we block here?

            int childCount = 1;
            foreach (var c in counts)
                childCount += c;

            // Check mesh buffer allocation.
            // Allocate morts for the parent level.
            Allocate(ref _gpuMesh, childCount, false);
            Allocate(ref _gpuParentMorts, childCount, false);

            // Build the bottom mesh level. Also computes morts for the parent level.
            _buildBottomKernel.SetMemoryArgument(0, _gpuInBodies);
            _buildBottomKernel.SetMemoryArgument(1, _gpuMesh);
            _buildBottomKernel.SetValueArgument(2, childCount);
            _buildBottomKernel.SetValueArgument(3, _bodies.Length);
            _buildBottomKernel.SetMemoryArgument(4, _gpuMapFlat);
            _buildBottomKernel.SetValueArgument(5, cellSizeExp);
            _buildBottomKernel.SetValueArgument(6, (int)Math.Pow(2.0f, cellSizeExp));
            _buildBottomKernel.SetMemoryArgument(7, _gpuParentMorts);
            _queue.Execute(_buildBottomKernel, null, new long[] { BlockCount(childCount) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);

            // Now build the top levels of the mesh.
            for (int level = 1; level <= _levels; level++)
            {
                // Clear count buffer.
                _queue.FillBuffer(_gpuCounts, new int[] { 0 }, 0, blocks, null);

                // Compute block count from child level cell counts.
                blocks = BlockCount(childCount);

                // Record locations of each mesh level.
                _levelIdx[level] = _levelIdx[level - 1] + childCount;

                // Build initial map from the morts computed at the child level.
                _cellMapKernel.SetMemoryArgument(0, _gpuParentMorts);
                _cellMapKernel.SetValueArgument(1, childCount);
                _cellMapKernel.SetMemoryArgument(2, _gpuMap);
                _cellMapKernel.SetMemoryArgument(3, _gpuCounts);
                _cellMapKernel.SetLocalArgument(4, SIZEOFINT * _threadsPerBlock);
                _cellMapKernel.SetValueArgument(5, 1); // Set step size to 1 for long input type.
                _queue.Execute(_cellMapKernel, null, new long[] { blocks * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);

                // Compress the cell map.
                _compressCellMapKernel.SetValueArgument(0, blocks);
                _compressCellMapKernel.SetMemoryArgument(1, _gpuMap);
                _compressCellMapKernel.SetMemoryArgument(2, _gpuMapFlat);
                _compressCellMapKernel.SetMemoryArgument(3, _gpuCounts);
                _queue.Execute(_compressCellMapKernel, null, new long[] { BlockCount(blocks) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);

                // Read and compute parent level cell count;
                counts = ReadBuffer(_gpuCounts, 0, blocks, false); // Should we block here?

                int parentCellCount = 1;
                foreach (var c in counts)
                    parentCellCount += c;

                // Record total mesh length.
                _meshLength = _levelIdx[level] + parentCellCount;

                // Make sure the mesh buffer is large enough.
                // Resize the buffer with the copy flag so that existing data isn't lost. (Hopefully...)
                Allocate(ref _gpuMesh, _meshLength, exact: false, copy: true);

                // Build the parent level. Also computes morts for the parents parent level.
                _buildTopKernel.SetMemoryArgument(0, _gpuMesh);
                _buildTopKernel.SetValueArgument(1, parentCellCount);
                _buildTopKernel.SetValueArgument(2, _levelIdx[level - 1]);
                _buildTopKernel.SetValueArgument(3, _levelIdx[level]);
                _buildTopKernel.SetMemoryArgument(4, _gpuMapFlat);
                _buildTopKernel.SetValueArgument(5, (int)Math.Pow(2.0f, cellSizeExp + level));
                _buildTopKernel.SetValueArgument(6, level);
                _buildTopKernel.SetMemoryArgument(7, _gpuParentMorts);
                _queue.Execute(_buildTopKernel, null, new long[] { BlockCount(parentCellCount) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);

                // Parent level now becomes a child...
                childCount = parentCellCount;
            }
        }

        /// <summary>
        /// Reads the sorted mort/index (long2) buffer and copies bodies to their sorted location.
        /// </summary>
        private void ReindexBodiesGPU()
        {
            _reindexKernel.SetMemoryArgument(0, _gpuOutBodies);
            _reindexKernel.SetValueArgument(1, _bodies.Length);
            _reindexKernel.SetMemoryArgument(2, _gpuBodyMorts);
            _reindexKernel.SetMemoryArgument(3, _gpuInBodies);
            _queue.Execute(_reindexKernel, null, new long[] { BlockCount(_bodies.Length) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
        }

        private void PopNeighborsMeshGPU(int meshSize)
        {
            // Calulate total size of 1D mesh neighbor list.
            // Each cell can have a max of 9 neighbors, including itself.
            int topSize = meshSize - _levelIdx[1];
            int neighborLen = topSize * 9;

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

        /// <summary>
        /// Allocates the specified buffer to the specified number of elements if needed.
        /// </summary>
        /// <typeparam name="T">Buffer type.</typeparam>
        /// <param name="buffer">The buffer to be reallocated.</param>
        /// <param name="count">The minimum number of elements the buffer should contain.</param>
        /// <param name="exact">If true, the buffer will be sized to exactly the specified number of elements.</param>
        /// <param name="copy">If true, the data from the old buffer will be copied to the new resized buffer.</param>
        /// <returns>The size in elements of the (potentially new) buffer.</returns>
        private long Allocate<T>(ref ComputeBuffer<T> buffer, long count, bool exact = false, bool copy = false) where T : struct
        {
            // Compute the max count based on the max allocation and type sizes.
            long typeSize = Marshal.SizeOf<T>();
            long maxCount = (_maxBufferSize / typeSize);

            // Record the current flags and count.
            var flags = buffer.Flags;
            long newCount = buffer.Count;

            // If the buffer will need resized.
            bool resizeNeeded = false;

            if (!exact)
            {
                // Check if the buffer needs resized.
                if (buffer.Count < count && buffer.Count < maxCount)
                {
                    resizeNeeded = true;

                    // Compute the new count from the grow factor.
                    newCount = (long)(count * BUF_GROW_FACTOR);

                    // Clamp size to max allowed.
                    newCount = Math.Min(newCount, maxCount);
                }
            }
            else
            {
                // Check if the buffer needs resized.
                if (buffer.Count != count)
                {
                    resizeNeeded = true;

                    // Clamp size to max allowed.
                    newCount = Math.Min(count, maxCount);
                }
            }

            if (resizeNeeded)
            {
                if (copy)
                {
                    // Create a new buffer then copy the data from the old buffer.
                    var newBuf = new ComputeBuffer<T>(_context, flags, newCount);
                    _queue.CopyBuffer(buffer, newBuf, 0, 0, buffer.Count, null);

                    // Dispose the old buffer then change the reference to the new one.
                    buffer.Dispose();
                    buffer = newBuf;
                }
                else
                {
                    // Just dispose then create a new buffer.
                    buffer.Dispose();
                    buffer = new ComputeBuffer<T>(_context, flags, newCount);
                }
            }

            return newCount;
        }


        private T[] ReadBuffer<T>(ComputeBuffer<T> buffer, bool blocking = false) where T : struct
        {
            T[] buf = new T[buffer.Count];

            _queue.ReadFromBuffer(buffer, ref buf, blocking, null);
            if (blocking) _queue.Finish(); // This is probably redundant...

            return buf;
        }

        private T[] ReadBuffer<T>(ComputeBuffer<T> buffer, long offset, long length, bool blocking = false) where T : struct
        {
            T[] buf = new T[length - offset];

            _queue.ReadFromBuffer(buffer, ref buf, blocking, offset, 0, length - offset, null);
            if (blocking) _queue.Finish(); // This is probably redundant...

            return buf;
        }
    }
}