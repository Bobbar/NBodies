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
        private bool _hasUnifiedMemory = false;

        private const int SIZEOFINT = 4; // Size of 32-bit integer in bytes.
        private const float BUF_GROW_FACTOR = 1.4f; // Reallocated buffers will increase in size by this multiple.
        private const int MESH_SIZE_NS_CUTOFF = 120000; // Mesh sizes greater than this will use the mesh-based neighbor search instead of binary.

        private int[] _levelIdx = new int[0]; // Locations of each level within the 1D mesh array.
        private MeshCell[] _mesh = new MeshCell[0]; // 1D array of mesh cells. (Populated on GPU, and read for UI display only.)
        private int _meshLength = 0; // Total length of the 1D mesh array.
        private Body[] _bodies = new Body[0]; // Local reference for the current body array.

        private ManualResetEventSlim _meshRequested = new ManualResetEventSlim(true);

        private ComputeContext _context;
        private ComputeCommandQueue _queue;
        private ComputeDevice _device = null;
        private ComputeProgram _program;

        private ComputeKernel _forceKernel;
        private ComputeKernel _collisionSPHKernel;
        private ComputeKernel _collisionElasticKernel;
        private ComputeKernel _buildNeighborsMeshKernel;
        private ComputeKernel _buildNeighborsBinaryKernel;
        private ComputeKernel _fixOverlapKernel;
        private ComputeKernel _buildBottomKernel;
        private ComputeKernel _buildTopKernel;
        private ComputeKernel _calcCMKernel;
        private ComputeKernel _reindexKernel;
        private ComputeKernel _cellMapKernel;
        private ComputeKernel _compressCellMapKernel;
        private ComputeKernel _computeMortsKernel;
        private ComputeKernel _histogramKernel;
        private ComputeKernel _reorderKernel;
        private ComputeKernel _scanHistogramsKernel;
        private ComputeKernel _pastehistogramsKernel;

        private ComputeBuffer<int> _gpuMeshNeighbors;
        private ComputeBuffer<Body> _gpuInBodies;
        private ComputeBuffer<Body> _gpuOutBodies;
        private ComputeBuffer<Vector3> _gpuCM;
        private ComputeBuffer<int> _gpuPostNeeded;
        private ComputeBuffer<int> _gpuMap;
        private ComputeBuffer<int> _gpuMapFlat;
        private ComputeBuffer<int> _gpuCounts;
        private ComputeBuffer<long> _gpuParentMorts;
        private ComputeBuffer<long2> _gpuBodyMortsA;
        private ComputeBuffer<long2> _gpuBodyMortsB;
        private ComputeBuffer<int> _gpuLevelCounts;
        private ComputeBuffer<int> _gpuLevelIdx;
        private ComputeBuffer<int> _gpuHistogram;
        private ComputeBuffer<int> _gpuGlobSum;
        private ComputeBuffer<int> _gpuGlobSumTemp;
        private ComputeBuffer<int2> _gpuMeshIdxs;
        private ComputeBuffer<int2> _gpuMeshNBounds;
        private ComputeBuffer<int2> _gpuMeshBodyBounds;
        private ComputeBuffer<int2> _gpuMeshChildBounds;
        private ComputeBuffer<float4> _gpuMeshCMM;
        private ComputeBuffer<int4> _gpuMeshSPL;


        private Stopwatch timer = new Stopwatch();
        private Stopwatch timer2 = new Stopwatch();

        private long _currentFrame = 0;
        private long _lastMeshReadFrame = 0;
        private long _curBufferVersion = -1;

        public MeshCell[] CurrentMesh
        {
            get
            {
                // Only read mesh from GPU if it has changed.
                if (_lastMeshReadFrame != _currentFrame)
                {
                    // Block until the mesh gets read at the end of the frame.
                    _meshRequested.Reset();
                    _meshRequested.Wait(200);

                    _lastMeshReadFrame = _currentFrame;
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

            _hasUnifiedMemory = _device.HostUnifiedMemory;

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
                    options = $@"-cl-std=CL2.0 -cl-fast-relaxed-math -D FASTMATH";
                else
                    options = $"-cl-std=CL2.0";

                _program.Build(new[] { _device }, options, null, IntPtr.Zero);
            }
            catch (BuildProgramFailureComputeException ex)
            {
                string buildLog = _program.GetBuildLog(_device);
                System.IO.File.WriteAllText("build_error.txt", buildLog);
                Console.WriteLine(buildLog);
                throw;
            }

            var bins = System.Text.Encoding.UTF8.GetString(_program.Binaries[0]);
            File.WriteAllText(Environment.CurrentDirectory + "/Physics/Kernels.ptx", bins);

            Console.WriteLine(_program.GetBuildLog(_device));
            System.IO.File.WriteAllText("build_log.txt", _program.GetBuildLog(_device));

            _forceKernel = _program.CreateKernel("CalcForce");
            _collisionSPHKernel = _program.CreateKernel("SPHCollisions");
            _collisionElasticKernel = _program.CreateKernel("ElasticCollisions");
            _buildNeighborsMeshKernel = _program.CreateKernel("BuildNeighborsMesh");
            _buildNeighborsBinaryKernel = _program.CreateKernel("BuildNeighborsBinary");
            _fixOverlapKernel = _program.CreateKernel("FixOverlaps");
            _buildBottomKernel = _program.CreateKernel("BuildBottom");
            _buildTopKernel = _program.CreateKernel("BuildTop");
            _calcCMKernel = _program.CreateKernel("CalcCenterOfMass");
            _reindexKernel = _program.CreateKernel("ReindexBodies");
            _cellMapKernel = _program.CreateKernel("MapMorts");
            _compressCellMapKernel = _program.CreateKernel("CompressMap");
            _computeMortsKernel = _program.CreateKernel("ComputeMorts");
            _histogramKernel = _program.CreateKernel("histogram");
            _reorderKernel = _program.CreateKernel("reorder");
            _scanHistogramsKernel = _program.CreateKernel("scanhistograms");
            _pastehistogramsKernel = _program.CreateKernel("pastehistograms");

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

        private void InitBuffers()
        {
            _gpuMeshNeighbors = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMeshNeighbors, 10000);

            _gpuInBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuInBodies, 1, true);

            _gpuOutBodies = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuOutBodies, 1, true);

            _gpuMeshIdxs = new ComputeBuffer<int2>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMeshIdxs, 10000, true);

            _gpuMeshNBounds = new ComputeBuffer<int2>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMeshNBounds, 10000, true);

            _gpuMeshBodyBounds = new ComputeBuffer<int2>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMeshBodyBounds, 10000, true);

            _gpuMeshChildBounds = new ComputeBuffer<int2>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMeshChildBounds, 10000, true);

            _gpuMeshCMM = new ComputeBuffer<float4>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMeshCMM, 10000, true);

            _gpuMeshSPL = new ComputeBuffer<int4>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMeshSPL, 10000, true);

            _gpuPostNeeded = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuPostNeeded, 1, true);

            _gpuCM = new ComputeBuffer<Vector3>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuCM, 1, true);

            _gpuBodyMortsA = new ComputeBuffer<long2>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuBodyMortsA, 1, true);

            _gpuBodyMortsB = new ComputeBuffer<long2>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuBodyMortsB, 1, true);

            _gpuMap = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMap, 10000, true);

            _gpuMapFlat = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMapFlat, 10000, true);

            _gpuCounts = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuCounts, 1, true);

            _gpuParentMorts = new ComputeBuffer<long>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuParentMorts, 10000, true);

            _gpuLevelCounts = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuLevelCounts, 1, true);

            _gpuLevelIdx = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuLevelIdx, 1, true);

            _gpuHistogram = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuHistogram, 10000, true);

            _gpuGlobSum = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuGlobSum, 10000, true);

            _gpuGlobSumTemp = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuGlobSumTemp, 10000, true);
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

        public void CalcMovement(ref Body[] bodies, SimSettings sim, int threadsPerBlock, long bufferVersion, bool alwaysReadBack, out bool isPostNeeded)
        {
            _bodies = bodies;
            _threadsPerBlock = threadsPerBlock;
            _levels = sim.MeshLevels;

            // Allocate and start writing bodies to the GPU.
            Allocate(ref _gpuInBodies, _bodies.Length);
            Allocate(ref _gpuOutBodies, _bodies.Length);

            // Only write the bodies buffer if it has been changed by the host.
            if (_curBufferVersion != bufferVersion)
            {
                WriteBuffer(_bodies, _gpuOutBodies, 0, 0, _bodies.Length);
            }

            // Post process flag.
            // Set by kernels when host side post processing is needed. (Culled and/or fractured bodies present)
            int[] postNeeded = new int[1] { 0 };
            WriteBuffer(postNeeded, _gpuPostNeeded, 0, 0, postNeeded.Length);

            // Recompute SPH pre-calcs if needed.
            if (_kernelSize != sim.KernelSize)
            {
                _kernelSize = sim.KernelSize;
                PreCalcSPH(_kernelSize);
            }

            // Build the particle mesh, mesh index, and mesh neighbors index.
            BuildMesh(sim.CellSizeExponent);

            // Get start and end of top level mesh cells.
            int meshTopStart = _levelIdx[_levels];
            int meshTopEnd = _meshLength;

            // Calc number of thread blocks to fit the dataset.
            int threadBlocks = BlockCount(_bodies.Length);


            //var sz = Marshal.SizeOf(typeof(Cloo.Bindings.CLMemoryHandle));


            // Compute gravity and SPH forces for the near/local field.
            int argi = 0;
            MyOCLBinding.SetMemoryArgument(_forceKernel.Handle, argi++, _gpuInBodies);
            MyOCLBinding.SetValueArgument(_forceKernel.Handle, argi++, _bodies.Length);
            //_forceKernel.SetValueArgument(argi++, _bodies.Length);
            MyOCLBinding.SetMemoryArgument(_forceKernel.Handle, argi++, _gpuMeshIdxs);
            MyOCLBinding.SetMemoryArgument(_forceKernel.Handle, argi++, _gpuMeshNBounds);
            //MyOCLBinding.SetMemoryArgument(_forceKernel.Handle, argi++, _gpuMeshBodyBounds);
            //MyOCLBinding.SetMemoryArgument(_forceKernel.Handle, argi++, _gpuMeshChildBounds);
            //MyOCLBinding.SetMemoryArgument(_forceKernel.Handle, argi++, _gpuMeshCMM);
            //MyOCLBinding.SetMemoryArgument(_forceKernel.Handle, argi++, _gpuMeshSPL);
            //MyOCLBinding.SetMemoryArgument(_forceKernel.Handle, argi++, _gpuMeshNeighbors);
            //MyOCLBinding.SetValueArgument(_forceKernel.Handle, argi++, sim);
            //MyOCLBinding.SetValueArgument(_forceKernel.Handle, argi++, _preCalcs);
            //MyOCLBinding.SetValueArgument(_forceKernel.Handle, argi++, meshTopStart);
            //MyOCLBinding.SetValueArgument(_forceKernel.Handle, argi++, meshTopEnd);
            //MyOCLBinding.SetMemoryArgument(_forceKernel.Handle, argi++, _gpuPostNeeded);

            //MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _forceKernel.Handle, 1, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, 0, null, null);



            //_forceKernel.SetMemoryArgument(argi++, _gpuInBodies);
            //_forceKernel.SetValueArgument(argi++, _bodies.Length);
            //_forceKernel.SetMemoryArgument(argi++, _gpuMeshIdxs);
            //_forceKernel.SetMemoryArgument(argi++, _gpuMeshNBounds);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshBodyBounds);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshChildBounds);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshCMM);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshSPL);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _forceKernel.SetValueArgument(argi++, sim);
            _forceKernel.SetValueArgument(argi++, _preCalcs);
            _forceKernel.SetValueArgument(argi++, meshTopStart);
            _forceKernel.SetValueArgument(argi++, meshTopEnd);
            _forceKernel.SetMemoryArgument(argi++, _gpuPostNeeded);
            _queue.Execute(_forceKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);



            //var bods = ReadBuffer(_gpuInBodies);

            //var b = bods[5276];
            //Debug.WriteLine($"[Host] Dens: {bods[5276].Density}  Press: {bods[5276].Pressure}");

            //_queue.Finish();
            //timer.Restart();

            //for (int i = 0; i < 1000; i++)
            //{
            //    //_queue.Execute(_forceKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);

            //    MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _forceKernel.Handle, 1, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, 0, null, null);

            //    //MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _forceKernel.Handle, 1, null, OCTools.ConvertArray(new long[] { threadBlocks * threadsPerBlock }), OCTools.ConvertArray(new long[] { threadsPerBlock }), 0, null, null);


            //}


            ////_queue.Finish();
            //timer.Print("Force");

            // Compute elastic collisions.
            argi = 0;
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _collisionElasticKernel.SetValueArgument(argi++, _bodies.Length);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMeshSPL);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMeshNBounds);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMeshBodyBounds);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _collisionElasticKernel.SetValueArgument(argi++, Convert.ToInt32(sim.CollisionsOn));
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuPostNeeded);
            //_queue.Execute(_collisionElasticKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);
            MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _collisionElasticKernel.Handle, 1, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, 0, null, null);
           
            // Compute SPH forces/collisions.
            argi = 0;
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _collisionSPHKernel.SetValueArgument(argi++, _bodies.Length);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuOutBodies);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshNBounds);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshSPL);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshChildBounds);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshIdxs);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshBodyBounds);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuCM);
            _collisionSPHKernel.SetValueArgument(argi++, sim);
            _collisionSPHKernel.SetValueArgument(argi++, _preCalcs);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuPostNeeded);
            //_queue.Execute(_collisionSPHKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);
            MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _collisionSPHKernel.Handle, 1, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, 0, null, null);

            // Read back post needed bool.
            ReadBuffer(_gpuPostNeeded, ref postNeeded, 0, 0, 1);
            isPostNeeded = Convert.ToBoolean(postNeeded[0]);

            // Read back bodies as needed.
            if (isPostNeeded || alwaysReadBack || _curBufferVersion != bufferVersion)
                ReadBuffer(_gpuOutBodies, ref bodies, 0, 0, bodies.Length);

            // Increment frame count.
            _currentFrame++;
            _curBufferVersion = bufferVersion;

            // Check if we need to read the mesh.
            if (!_meshRequested.IsSet)
            {
                ReadMesh();
                _meshRequested.Set();
            }
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

                ReadBuffer(outBodies, ref bodies, 0, 0, bodies.Length);
                _queue.Finish();
            }
        }

        private void ReadMesh()
        {
            if (_mesh.Length != _meshLength)
                _mesh = new MeshCell[_meshLength];

            var meshIdxs = ReadBuffer(_gpuMeshIdxs);
            var meshNBounds = ReadBuffer(_gpuMeshNBounds);
            var meshBodyBounds = ReadBuffer(_gpuMeshBodyBounds);
            var meshChildBounds = ReadBuffer(_gpuMeshChildBounds);
            var meshCMM = ReadBuffer(_gpuMeshCMM);
            var meshSPL = ReadBuffer(_gpuMeshSPL);

            for (int i = 0; i < _meshLength; i++)
            {
                _mesh[i].IdxX = meshIdxs[i].X;
                _mesh[i].IdxY = meshIdxs[i].Y;
                _mesh[i].NeighborStartIdx = meshNBounds[i].X;
                _mesh[i].NeighborCount = meshNBounds[i].Y;
                _mesh[i].BodyStartIdx = meshBodyBounds[i].X;
                _mesh[i].BodyCount = meshBodyBounds[i].Y;
                _mesh[i].ChildStartIdx = meshChildBounds[i].X;
                _mesh[i].ChildCount = meshChildBounds[i].Y;
                _mesh[i].Mass = meshCMM[i].Z;
                _mesh[i].CmX = meshCMM[i].X;
                _mesh[i].CmY = meshCMM[i].Y;
                _mesh[i].Size = meshSPL[i].X;
                _mesh[i].ParentID = meshSPL[i].Y;
                _mesh[i].Level = meshSPL[i].Z;

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
        /// Builds the particle mesh and mesh-neighbor index for the current field.
        /// </summary>
        /// <param name="cellSizeExp">Cell size exponent. 2 ^ exponent = cell size.</param>
        private void BuildMesh(int cellSizeExp)
        {
            // Index of mesh level locations.
            _levelIdx = new int[_levels + 2];
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
            _calcCMKernel.SetMemoryArgument(0, _gpuMeshCMM);
            _calcCMKernel.SetMemoryArgument(1, _gpuCM);
            _calcCMKernel.SetValueArgument(2, _levelIdx[_levels]);
            _calcCMKernel.SetValueArgument(3, _meshLength);
            _queue.ExecuteTask(_calcCMKernel, null);

            // Build Nearest Neighbor List.
            if (_meshLength > MESH_SIZE_NS_CUTOFF)
                PopNeighborsMeshGPU(_meshLength);
            else
                PopNeighborsBinaryGPU(_meshLength);
        }

        private int ComputePaddedSize(int len)
        {
            // Radix sort input length must be divisible by this value.
            const int radixMulti = 1024;

            if (len < radixMulti)
                return radixMulti;

            int mod = len % radixMulti;
            int padLen = (len - mod) + radixMulti;
            return padLen;
        }

        private void ComputeMortsGPU(int padLen, int cellSizeExp)
        {
            Allocate(ref _gpuBodyMortsA, padLen, exact: true);

            _computeMortsKernel.SetMemoryArgument(0, _gpuOutBodies);
            _computeMortsKernel.SetValueArgument(1, _bodies.Length);
            _computeMortsKernel.SetValueArgument(2, padLen);
            _computeMortsKernel.SetValueArgument(3, cellSizeExp);
            _computeMortsKernel.SetMemoryArgument(4, _gpuBodyMortsA);
            //_queue.Execute(_computeMortsKernel, null, new long[] { BlockCount(padLen) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
            MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _computeMortsKernel.Handle, 1, null, new long[] { BlockCount(padLen) * _threadsPerBlock }, new long[] { _threadsPerBlock }, 0, null, null);
        }

        //
        // Credit: https://github.com/gyatskov/radix-sort
        // https://github.com/modelflat/OCLRadixSort
        //
        private void SortByMortGPU(int padLen)
        {
            // Radix constants.
            const int _NUM_BITS_PER_RADIX = 8;
            const int _NUM_ITEMS_PER_GROUP = 4;
            const int _NUM_GROUPS = 128;
            const int _RADIX = (1 << _NUM_BITS_PER_RADIX);
            const int _DATA_SIZE = 8;
            const int _NUM_HISTOSPLIT = 512;
            const int _TOTALBITS = _DATA_SIZE << 3;
            const int _NUM_PASSES = (_TOTALBITS / _NUM_BITS_PER_RADIX);

            int numItems = _NUM_ITEMS_PER_GROUP * _NUM_GROUPS;
            int numLocalItems = _NUM_ITEMS_PER_GROUP;

            Allocate(ref _gpuBodyMortsB, padLen, true);
            Allocate(ref _gpuHistogram, _RADIX * _NUM_GROUPS * _NUM_ITEMS_PER_GROUP, true);
            Allocate(ref _gpuGlobSum, _NUM_HISTOSPLIT, true);
            Allocate(ref _gpuGlobSumTemp, _NUM_HISTOSPLIT, true);

            for (int pass = 0; pass < _NUM_PASSES; pass++)
            {
                #region Histogram
                // Histogram
                int argi = 0;
                _histogramKernel.SetMemoryArgument(argi++, _gpuBodyMortsA);
                _histogramKernel.SetMemoryArgument(argi++, _gpuHistogram);
                _histogramKernel.SetValueArgument(argi++, pass);
                _histogramKernel.SetLocalArgument(argi++, SIZEOFINT * _RADIX * _NUM_ITEMS_PER_GROUP);
                _histogramKernel.SetValueArgument(argi++, padLen);
                //_queue.Execute(_histogramKernel, null, new long[] { numItems }, new long[] { numLocalItems }, null);
                MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _histogramKernel.Handle, 1, null, new long[] { numItems }, new long[] { numLocalItems }, 0, null, null);

                //
                #endregion Histogram

                #region Scan
                // Scan

                // Pass 1
                numItems = _RADIX * _NUM_GROUPS * _NUM_ITEMS_PER_GROUP / 2;
                numLocalItems = numItems / _NUM_HISTOSPLIT;

                argi = 0;
                _scanHistogramsKernel.SetMemoryArgument(argi++, _gpuHistogram);
                _scanHistogramsKernel.SetLocalArgument(argi++, SIZEOFINT * _NUM_HISTOSPLIT);
                _scanHistogramsKernel.SetMemoryArgument(argi++, _gpuGlobSum);
                //_queue.Execute(_scanHistogramsKernel, null, new long[] { numItems }, new long[] { numLocalItems }, null);
                MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _scanHistogramsKernel.Handle, 1, null, new long[] { numItems }, new long[] { numLocalItems }, 0, null, null);

                // Pass 2
                numItems = _NUM_HISTOSPLIT / 2;
                numLocalItems = numItems;

                _scanHistogramsKernel.SetMemoryArgument(0, _gpuGlobSum);
                _scanHistogramsKernel.SetMemoryArgument(2, _gpuGlobSumTemp);
                //_queue.Execute(_scanHistogramsKernel, null, new long[] { numItems }, new long[] { numLocalItems }, null);
                MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _scanHistogramsKernel.Handle, 1, null, new long[] { numItems }, new long[] { numLocalItems }, 0, null, null);

                #endregion Scan

                #region Merge
                // Merge Histograms
                numItems = _RADIX * _NUM_GROUPS * _NUM_ITEMS_PER_GROUP / 2;
                numLocalItems = numItems / _NUM_HISTOSPLIT;

                _pastehistogramsKernel.SetMemoryArgument(0, _gpuHistogram);
                _pastehistogramsKernel.SetMemoryArgument(1, _gpuGlobSum);
                //_queue.Execute(_pastehistogramsKernel, null, new long[] { numItems }, new long[] { numLocalItems }, null);
                MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _pastehistogramsKernel.Handle, 1, null, new long[] { numItems }, new long[] { numLocalItems }, 0, null, null);

                #endregion Merge

                #region Reorder
                // Reorder
                numLocalItems = _NUM_ITEMS_PER_GROUP;
                numItems = _NUM_ITEMS_PER_GROUP * _NUM_GROUPS;

                argi = 0;
                _reorderKernel.SetMemoryArgument(argi++, _gpuBodyMortsA);
                _reorderKernel.SetMemoryArgument(argi++, _gpuBodyMortsB);
                _reorderKernel.SetMemoryArgument(argi++, _gpuHistogram);
                _reorderKernel.SetValueArgument(argi++, pass);
                _reorderKernel.SetLocalArgument(argi++, SIZEOFINT * _RADIX * _NUM_ITEMS_PER_GROUP);
                _reorderKernel.SetValueArgument(argi++, padLen);
                //_queue.Execute(_reorderKernel, null, new long[] { numItems }, new long[] { numLocalItems }, null);
                MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _reorderKernel.Handle, 1, null, new long[] { numItems }, new long[] { numLocalItems }, 0, null, null);

                #endregion Reorder

                // Swap keys buffers.
                ComputeBuffer<long2> temp = _gpuBodyMortsA;
                _gpuBodyMortsA = _gpuBodyMortsB;
                _gpuBodyMortsB = temp;
            }
        }

        /// <summary>
        /// Reads the sorted mort/index (long2) buffer and copies bodies to their sorted location.
        /// </summary>
        private void ReindexBodiesGPU()
        {
            // This kernel is a bit faster with less threads per block... (Why?)
            int threads = 8;
            _reindexKernel.SetMemoryArgument(0, _gpuOutBodies);
            _reindexKernel.SetValueArgument(1, _bodies.Length);
            _reindexKernel.SetMemoryArgument(2, _gpuBodyMortsA);
            _reindexKernel.SetMemoryArgument(3, _gpuInBodies);
            //_queue.Execute(_reindexKernel, null, new long[] { BlockCount(_bodies.Length, threads) * threads }, new long[] { threads }, null);
            MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _reindexKernel.Handle, 1, null, new long[] { BlockCount(_bodies.Length, threads) * threads }, new long[] { threads }, 0, null, null);

        }

        private void BuildMeshGPU(int cellSizeExp)
        {
            // First level map is built from body morts.
            // Subsequent parent level maps are built from child cell morts.

            // Compute the block count and workgroup sizes from the # of bodies.
            int blocks = BlockCount(_bodies.Length);
            long[] globalSize = new long[] { blocks * _threadsPerBlock };
            long[] globalSizeComp = new long[] { BlockCount(blocks) * _threadsPerBlock };
            long[] localSize = new long[] { _threadsPerBlock };

            // Allocate level counts & index.
            Allocate(ref _gpuLevelCounts, _levels + 1, true);
            Allocate(ref _gpuLevelIdx, _levels + 2, true);

            // Clear the level index.
            _queue.FillBuffer(_gpuLevelIdx, new int[1] { 0 }, 0, _levels + 2, null);

            // Allocate map and count buffers.
            Allocate(ref _gpuMap, _bodies.Length, false);
            Allocate(ref _gpuMapFlat, _bodies.Length, false);
            Allocate(ref _gpuCounts, blocks, false);

            // Allocate mesh and morts buffers to body count.
            long bufLen = Allocate(ref _gpuMeshIdxs, _bodies.Length, false);
            Allocate(ref _gpuMeshNBounds, _bodies.Length, false);
            Allocate(ref _gpuMeshBodyBounds, _bodies.Length, false);
            Allocate(ref _gpuMeshChildBounds, _bodies.Length, false);
            Allocate(ref _gpuMeshCMM, _bodies.Length, false);
            Allocate(ref _gpuMeshSPL, _bodies.Length, false);
            Allocate(ref _gpuParentMorts, _bodies.Length, false);

            // Build initial map from sorted body morts.
            _cellMapKernel.SetMemoryArgument(0, _gpuBodyMortsA);
            _cellMapKernel.SetValueArgument(1, _bodies.Length);
            _cellMapKernel.SetMemoryArgument(2, _gpuMap);
            _cellMapKernel.SetMemoryArgument(3, _gpuCounts);
            _cellMapKernel.SetLocalArgument(4, SIZEOFINT * _threadsPerBlock);
            _cellMapKernel.SetValueArgument(5, 2); // Set step size to 2 for long2 input type.
            _cellMapKernel.SetValueArgument(6, 0);
            _cellMapKernel.SetMemoryArgument(7, _gpuLevelCounts);
            _cellMapKernel.SetValueArgument(8, _gpuBodyMortsA.Count);
            //_queue.Execute(_cellMapKernel, null, globalSize, localSize, null);
            MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _cellMapKernel.Handle, 1, null, globalSize, localSize, 0, null, null);

            // Remove the gaps to compress the cell map into the beginning of the buffer.
            // This allows the map to be read properly by the mesh building kernels.
            _compressCellMapKernel.SetValueArgument(0, blocks);
            _compressCellMapKernel.SetMemoryArgument(1, _gpuMap);
            _compressCellMapKernel.SetMemoryArgument(2, _gpuMapFlat);
            _compressCellMapKernel.SetMemoryArgument(3, _gpuCounts);
            _compressCellMapKernel.SetMemoryArgument(4, _gpuLevelCounts);
            _compressCellMapKernel.SetMemoryArgument(5, _gpuLevelIdx);
            _compressCellMapKernel.SetValueArgument(6, 0);
            //_queue.Execute(_compressCellMapKernel, null, globalSizeComp, localSize, null);
            MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _compressCellMapKernel.Handle, 1, null, globalSizeComp, localSize, 0, null, null);

            // Build the bottom mesh level. Also computes morts for the parent level.
            int argi = 0;
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuInBodies);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuMeshIdxs);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuMeshBodyBounds);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuMeshCMM);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuMeshSPL);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuLevelCounts);
            _buildBottomKernel.SetValueArgument(argi++, _bodies.Length);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuMapFlat);
            _buildBottomKernel.SetValueArgument(argi++, cellSizeExp);
            _buildBottomKernel.SetValueArgument(argi++, (int)Math.Pow(2.0f, cellSizeExp));
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuParentMorts);
            _buildBottomKernel.SetValueArgument(argi++, bufLen);
            //_queue.Execute(_buildBottomKernel, null, globalSize, localSize, null);
            MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _buildBottomKernel.Handle, 1, null, globalSize, localSize, 0, null, null);


            // Now build the top levels of the mesh.
            // NOTE: We use the same kernel work sizes as the bottom level,
            // but kernels outside the scope of work will just return and idle.
            for (int level = 1; level <= _levels; level++)
            {
                // Build initial map from the morts computed at the child level.
                _cellMapKernel.SetMemoryArgument(0, _gpuParentMorts);
                _cellMapKernel.SetValueArgument(1, -1); // We don't know the length, so set it to -1 to make the kernel read it from the level counts buffer.
                _cellMapKernel.SetMemoryArgument(2, _gpuMap);
                _cellMapKernel.SetMemoryArgument(3, _gpuCounts);
                _cellMapKernel.SetLocalArgument(4, SIZEOFINT * _threadsPerBlock);
                _cellMapKernel.SetValueArgument(5, 1); // Set step size to 1 for long input type.
                _cellMapKernel.SetValueArgument(6, level);
                _cellMapKernel.SetMemoryArgument(7, _gpuLevelCounts);
                _cellMapKernel.SetValueArgument(8, _gpuParentMorts.Count);
                //_queue.Execute(_cellMapKernel, null, globalSize, localSize, null);
                MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _cellMapKernel.Handle, 1, null, globalSize, localSize, 0, null, null);

                // Compress the cell map.
                _compressCellMapKernel.SetValueArgument(0, -1); // Same as above. Make the kernel read length from the level counts buffer.
                _compressCellMapKernel.SetMemoryArgument(1, _gpuMap);
                _compressCellMapKernel.SetMemoryArgument(2, _gpuMapFlat);
                _compressCellMapKernel.SetMemoryArgument(3, _gpuCounts);
                _compressCellMapKernel.SetMemoryArgument(4, _gpuLevelCounts);
                _compressCellMapKernel.SetMemoryArgument(5, _gpuLevelIdx);
                _compressCellMapKernel.SetValueArgument(6, level);
                //_queue.Execute(_compressCellMapKernel, null, globalSizeComp, localSize, null);
                MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _compressCellMapKernel.Handle, 1, null, globalSizeComp, localSize, 0, null, null);

                // Build the parent level. Also computes morts for the parents parent level.
                argi = 0;
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMeshIdxs);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMeshBodyBounds);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMeshChildBounds);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMeshCMM);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMeshSPL);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuLevelCounts);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuLevelIdx);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMapFlat);
                _buildTopKernel.SetValueArgument(argi++, (int)Math.Pow(2.0f, cellSizeExp + level));
                _buildTopKernel.SetValueArgument(argi++, level);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuParentMorts);
                _buildTopKernel.SetValueArgument(argi++, bufLen);
                //_queue.Execute(_buildTopKernel, null, globalSize, localSize, null);
                MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _buildTopKernel.Handle, 1, null, globalSize, localSize, 0, null, null);

            }

            // Read back the level index and set the total mesh length.
            ReadBuffer(_gpuLevelIdx, ref _levelIdx, 0, 0, _levelIdx.Length);
            _meshLength = _levelIdx[_levels + 1];

            // If the mesh buffer was too small, reallocate and rebuild it again.
            // This done because we are not reading back counts for each level and reallocating,
            // so we don't know if we have enough room until a build has completed.
            if (bufLen < _meshLength)
            {
                Debug.WriteLine($"Mesh reallocated: {bufLen} -> {_meshLength}");

                Allocate(ref _gpuMeshIdxs, _meshLength);
                Allocate(ref _gpuMeshNBounds, _meshLength);
                Allocate(ref _gpuMeshBodyBounds, _meshLength);
                Allocate(ref _gpuMeshChildBounds, _meshLength);
                Allocate(ref _gpuMeshCMM, _meshLength);
                Allocate(ref _gpuMeshSPL, _meshLength);
                Allocate(ref _gpuParentMorts, _meshLength);
                Allocate(ref _gpuMap, _meshLength);
                Allocate(ref _gpuMapFlat, _meshLength);

                BuildMeshGPU(cellSizeExp);
            }
        }

        /// <summary>
        /// Neighbor search using a binary search strategy. Faster with smaller mesh cell count.
        /// </summary>
        private void PopNeighborsBinaryGPU(int meshSize)
        {
            int topSize = meshSize - _levelIdx[1];
            int neighborLen = topSize * 9;

            Allocate(ref _gpuMeshNeighbors, neighborLen);

            int workSize = BlockCount(topSize) * _threadsPerBlock;

            int argi = 0;
            _buildNeighborsBinaryKernel.SetMemoryArgument(argi++, _gpuMeshIdxs);
            _buildNeighborsBinaryKernel.SetMemoryArgument(argi++, _gpuMeshNBounds);
            _buildNeighborsBinaryKernel.SetMemoryArgument(argi++, _gpuMeshSPL);
            _buildNeighborsBinaryKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            _buildNeighborsBinaryKernel.SetMemoryArgument(argi++, _gpuLevelIdx);
            _buildNeighborsBinaryKernel.SetValueArgument(argi++, topSize);
            _buildNeighborsBinaryKernel.SetValueArgument(argi++, _levelIdx[1]);
            //_queue.Execute(_buildNeighborsBinaryKernel, null, new long[] { workSize }, new long[] { _threadsPerBlock }, null);
            MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _buildNeighborsBinaryKernel.Handle, 1, null, new long[] { workSize }, new long[] { _threadsPerBlock }, 0, null, null);

        }

        /// <summary>
        /// Neighbor search using top-down mesh/tree hierarchy strategy. Faster with larger mesh cell count.
        /// </summary>
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
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMeshIdxs);
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMeshSPL);
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMeshNBounds);
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMeshChildBounds);
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, _levelIdx[1]);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, _levels);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, level);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, start);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, end);
                //_queue.Execute(_buildNeighborsMeshKernel, null, new long[] { workSize }, new long[] { _threadsPerBlock }, null);
                MyOCLBinding.EnqueueNDRangeKernel(_queue.Handle, _buildNeighborsMeshKernel.Handle, 1, null, new long[] { workSize }, new long[] { _threadsPerBlock }, 0, null, null);

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

            if (_hasUnifiedMemory) 
                blocking = true;

            _queue.ReadFromBuffer(buffer, ref buf, blocking, null);
            if (blocking) _queue.Finish(); // This is probably redundant...

            return buf;
        }

        private T[] ReadBuffer<T>(ComputeBuffer<T> buffer, long offset, long length, bool blocking = false) where T : struct
        {
            T[] buf = new T[length - offset];

            if (_hasUnifiedMemory)
                blocking = true;

            _queue.ReadFromBuffer(buffer, ref buf, blocking, offset, 0, length - offset, null);
            if (blocking) _queue.Finish(); // This is probably redundant...

            return buf;
        }

        private void ReadBuffer<T>(ComputeBufferBase<T> source, ref T[] dest, long sourceOffset, long destOffset, long region) where T : struct
        {
            var sizeofT = Marshal.SizeOf<T>();
            GCHandle destinationGCHandle = GCHandle.Alloc(dest, GCHandleType.Pinned);
            IntPtr destinationOffsetPtr = Marshal.UnsafeAddrOfPinnedArrayElement(dest, (int)destOffset);

            bool blocking = false;
            if (_hasUnifiedMemory)
                blocking = true;

            MyOCLBinding.EnqueueReadBuffer(_queue.Handle,
                source.Handle,
                blocking,
                new IntPtr(0 * sizeofT),
                new IntPtr(dest.Length * sizeofT),
                destinationOffsetPtr,
                0,
                null,
                null);
          
                //Cloo.Bindings.CL12.EnqueueReadBuffer(
                //_queue.Handle,
                //source.Handle,
                //blocking,
                //new IntPtr(0 * sizeofT),
                //new IntPtr(dest.Length * sizeofT),
                //destinationOffsetPtr,
                //0,
                //null,
                //null);

            destinationGCHandle.Free();
        }

        private void WriteBuffer<T>(T[] source, ComputeBufferBase<T> dest, long sourceOffset, long destOffset, long region) where T : struct
        {
            var sizeofT = Marshal.SizeOf<T>();
            GCHandle sourceGCHandle = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr sourceOffsetPtr = Marshal.UnsafeAddrOfPinnedArrayElement(source, (int)sourceOffset);

            bool blocking = false;
            if (_hasUnifiedMemory)
                blocking = true;

            Cloo.Bindings.CL12.EnqueueWriteBuffer(
                _queue.Handle,
                dest.Handle,
                blocking,
                new IntPtr(0 * sizeofT),
                new IntPtr(source.Length * sizeofT),
                sourceOffsetPtr,
                0,
                null,
                null);

            sourceGCHandle.Free();
        }

        public void Flush()
        {
            _curBufferVersion = int.MaxValue;

            _meshRequested.Set();
            _meshLength = 0;
            _gpuMeshNeighbors.Dispose();
            _gpuInBodies.Dispose();
            _gpuOutBodies.Dispose();
            _gpuCM.Dispose();
            _gpuPostNeeded.Dispose();

            _gpuCounts.Dispose();
            _gpuParentMorts.Dispose();
            _gpuMap.Dispose();
            _gpuMapFlat.Dispose();
            _gpuBodyMortsA.Dispose();
            _gpuBodyMortsB.Dispose();
            _gpuLevelCounts.Dispose();
            _gpuLevelIdx.Dispose();
            _gpuHistogram.Dispose();
            _gpuGlobSum.Dispose();
            _gpuGlobSumTemp.Dispose();
            _gpuMeshIdxs.Dispose();
            _gpuMeshNBounds.Dispose();
            _gpuMeshBodyBounds.Dispose();
            _gpuMeshChildBounds.Dispose();
            _gpuMeshCMM.Dispose();
            _gpuMeshSPL.Dispose();

            _mesh = new MeshCell[0];
            _currentFrame = 0;
            _lastMeshReadFrame = 0;
            InitBuffers();
        }

        public void Dispose()
        {
            _gpuMeshNeighbors.Dispose();
            _gpuInBodies.Dispose();
            _gpuOutBodies.Dispose();
            _gpuCM.Dispose();
            _gpuPostNeeded.Dispose();
            _gpuCounts.Dispose();
            _gpuParentMorts.Dispose();
            _gpuMap.Dispose();
            _gpuMapFlat.Dispose();
            _gpuBodyMortsA.Dispose();
            _gpuLevelCounts.Dispose();
            _gpuLevelIdx.Dispose();
            _gpuBodyMortsB.Dispose();
            _gpuHistogram.Dispose();
            _gpuGlobSum.Dispose();
            _gpuGlobSumTemp.Dispose();
            _gpuMeshIdxs.Dispose();
            _gpuMeshNBounds.Dispose();
            _gpuMeshBodyBounds.Dispose();
            _gpuMeshChildBounds.Dispose();
            _gpuMeshCMM.Dispose();
            _gpuMeshSPL.Dispose();



            _forceKernel.Dispose();
            _collisionSPHKernel.Dispose();
            _collisionElasticKernel.Dispose();
            _buildNeighborsMeshKernel.Dispose();
            _buildNeighborsBinaryKernel.Dispose();
            _fixOverlapKernel.Dispose();
            _buildBottomKernel.Dispose();
            _buildTopKernel.Dispose();
            _calcCMKernel.Dispose();
            _reindexKernel.Dispose();
            _cellMapKernel.Dispose();
            _compressCellMapKernel.Dispose();
            _computeMortsKernel.Dispose();
            _histogramKernel.Dispose();
            _reorderKernel.Dispose();
            _scanHistogramsKernel.Dispose();
            _pastehistogramsKernel.Dispose();

            _program.Dispose();
            _context.Dispose();
            _queue.Dispose();
        }
    }
}