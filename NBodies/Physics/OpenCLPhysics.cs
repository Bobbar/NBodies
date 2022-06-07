﻿using Cloo;
using Cloo.Bindings;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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
        private bool _profile = false;
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
        private int[] _levelIdx = new int[1]; // Locations of each level within the 1D mesh array.
        private MeshCell[] _mesh = new MeshCell[0]; // 1D array of mesh cells. (Populated on GPU, and read for UI display only.)
        private int _meshLength = 0; // Total length of the 1D mesh array.
        private Body[] _bodies = new Body[1]; // Local reference for the current body array.
        private int[] _levelCounts = new int[1];

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
        private ComputeKernel _cellMapKernel;
        private ComputeKernel _compressCellMapKernel;
        private ComputeKernel _computeMortsKernel;
        private ComputeKernel _histogramKernel;
        private ComputeKernel _reorderKernel;
        private ComputeKernel _scanHistogramsKernel;
        private ComputeKernel _pastehistogramsKernel;


        private ComputeBuffer<Body>[] _gpuBodies = new ComputeBuffer<Body>[2];
        private ComputeBuffer<float2> _gpuCM;
        private ComputeBuffer<int> _gpuPostNeeded;
        private ComputeBuffer<int> _gpuMap;
        private ComputeBuffer<int> _gpuMapFlat;
        private ComputeBuffer<int> _gpuCounts;
        private ComputeBuffer<long> _gpuParentMorts;
        private ComputeBuffer<long2>[] _gpuBodyMorts = new ComputeBuffer<long2>[2];
        private ComputeBuffer<int> _gpuLevelCounts;
        private ComputeBuffer<int> _gpuLevelIdx;
        private ComputeBuffer<int> _gpuHistogram;
        private ComputeBuffer<int> _gpuGlobSum;
        private ComputeBuffer<int> _gpuGlobSumTemp;

        // GPU mesh buffers.
        private MeshGpuBuffers _gpuMeshBufs = new MeshGpuBuffers();

        // Host mesh buffers.
        private MeshHostBuffers _hostMeshBufs = new MeshHostBuffers();


        private Stopwatch timer = new Stopwatch();
        private ComputeEventList _events = null;
        private float _bestProfileElap = float.MaxValue;

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

        public OpenCLPhysics(ComputeDevice device, int threadsperblock, bool fastMath)
        {
            _device = device;
            _useFastMath = fastMath;

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

            var flag = ComputeCommandQueueFlags.None;
            if (_profile)
                flag = ComputeCommandQueueFlags.Profiling;

            _queue = new ComputeCommandQueue(_context, _device, flag);

            var kernelPath = $@"{Environment.CurrentDirectory}\Physics\Kernels\";

            StreamReader streamReader = new StreamReader($@"{kernelPath}\Physics.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            _program = new ComputeProgram(_context, clSource);

            try
            {
                string options = $@"-cl-std=CL2.0 -I {kernelPath} ";

                if (_useFastMath)
                    options += "-cl-fast-relaxed-math";

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

            InitKernels();
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

        private void InitKernels()
        {
            _forceKernel = _program.CreateKernel("CalcForce");
            _collisionSPHKernel = _program.CreateKernel("SPHCollisions");
            _collisionElasticKernel = _program.CreateKernel("ElasticCollisions");
            _buildNeighborsMeshKernel = _program.CreateKernel("BuildNeighborsMesh");
            _buildNeighborsBinaryKernel = _program.CreateKernel("BuildNeighborsBinary");
            _fixOverlapKernel = _program.CreateKernel("FixOverlaps");
            _buildBottomKernel = _program.CreateKernel("BuildBottom");
            _buildTopKernel = _program.CreateKernel("BuildTop");
            _calcCMKernel = _program.CreateKernel("CalcCenterOfMass");
            _cellMapKernel = _program.CreateKernel("MapMorts");
            _compressCellMapKernel = _program.CreateKernel("CompressMap");
            _computeMortsKernel = _program.CreateKernel("ComputeMorts");
            _histogramKernel = _program.CreateKernel("histogram");
            _reorderKernel = _program.CreateKernel("reorder");
            _scanHistogramsKernel = _program.CreateKernel("scanhistograms");
            _pastehistogramsKernel = _program.CreateKernel("pastehistograms");
        }

        private void InitBuffers()
        {
            _gpuBodies[0] = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, _bodies);
            _gpuBodies[1] = new ComputeBuffer<Body>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, _bodies);

            _gpuMeshBufs.Indexes = new ComputeBuffer<int2>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, _hostMeshBufs.Indexes);
            _gpuMeshBufs.NeighborBounds = new ComputeBuffer<int2>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, _hostMeshBufs.NeighborBounds);
            _gpuMeshBufs.BodyBounds = new ComputeBuffer<int2>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, _hostMeshBufs.BodyBounds);
            _gpuMeshBufs.ChildBounds = new ComputeBuffer<int2>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, _hostMeshBufs.ChildBounds);
            _gpuMeshBufs.CenterMass = new ComputeBuffer<float4>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, _hostMeshBufs.CenterMass);
            _gpuMeshBufs.SizeParentLevel = new ComputeBuffer<int4>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, _hostMeshBufs.SizeParentLevel);
            _gpuMeshBufs.Neighbors = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMeshBufs.Neighbors, 10000);

            _gpuPostNeeded = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuPostNeeded, 1, true);

            _gpuCM = new ComputeBuffer<float2>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuCM, 1, true);

            _gpuBodyMorts[0] = new ComputeBuffer<long2>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuBodyMorts[0], 1, true);

            _gpuBodyMorts[1] = new ComputeBuffer<long2>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuBodyMorts[1], 1, true);

            _gpuMap = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMap, 10000, true);

            _gpuMapFlat = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuMapFlat, 10000, true);

            _gpuCounts = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuCounts, 1, true);

            _gpuParentMorts = new ComputeBuffer<long>(_context, ComputeMemoryFlags.ReadWrite, 1);
            Allocate(ref _gpuParentMorts, 10000, true);

            _gpuLevelCounts = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, _levelCounts);
            _gpuLevelIdx = new ComputeBuffer<int>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, _levelIdx);

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
            if (_profile)
                _events = new ComputeEventList();

            _bodies = bodies;
            _threadsPerBlock = threadsPerBlock;
            _levels = sim.MeshLevels;

            // Only write the bodies buffer if it has been changed by the host.
            if (_curBufferVersion != bufferVersion)
            {
                Allocate(ref _gpuBodies[0], _bodies);
                Allocate(ref _gpuBodies[1], _bodies);
            }

            // Post process flag.
            // Set by kernels when host side post processing is needed. (Culled and/or fractured bodies present)
            int[] postNeeded = new int[1] { 0 };
            WriteBuffer(postNeeded, _gpuPostNeeded, 0, 0, postNeeded.Length, _events);

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

            // Compute gravity and SPH forces for the near/local field.
            int argi = 0;
            _forceKernel.SetMemoryArgument(argi++, _gpuBodies[0]);
            _forceKernel.SetValueArgument(argi++, _bodies.Length);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshBufs.Indexes);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshBufs.NeighborBounds);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshBufs.BodyBounds);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshBufs.ChildBounds);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshBufs.CenterMass);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshBufs.SizeParentLevel);
            _forceKernel.SetMemoryArgument(argi++, _gpuMeshBufs.Neighbors);
            _forceKernel.SetValueArgument(argi++, sim);
            _forceKernel.SetValueArgument(argi++, _preCalcs);
            _forceKernel.SetValueArgument(argi++, meshTopStart);
            _forceKernel.SetValueArgument(argi++, meshTopEnd);
            _forceKernel.SetMemoryArgument(argi++, _gpuPostNeeded);
            _queue.Execute(_forceKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, _events);

            // Compute elastic collisions.
            argi = 0;
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuBodies[0]);
            _collisionElasticKernel.SetValueArgument(argi++, _bodies.Length);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMeshBufs.SizeParentLevel);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMeshBufs.NeighborBounds);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMeshBufs.BodyBounds);
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuMeshBufs.Neighbors);
            _collisionElasticKernel.SetValueArgument(argi++, Convert.ToInt32(sim.CollisionsOn));
            _collisionElasticKernel.SetMemoryArgument(argi++, _gpuPostNeeded);
            _queue.Execute(_collisionElasticKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, _events);

            // Compute SPH forces/collisions.
            argi = 0;
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuBodies[0]);
            _collisionSPHKernel.SetValueArgument(argi++, _bodies.Length);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuBodies[1]);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshBufs.NeighborBounds);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshBufs.SizeParentLevel);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshBufs.ChildBounds);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshBufs.Indexes);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshBufs.BodyBounds);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuMeshBufs.Neighbors);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuCM);
            _collisionSPHKernel.SetValueArgument(argi++, sim);
            _collisionSPHKernel.SetValueArgument(argi++, _preCalcs);
            _collisionSPHKernel.SetMemoryArgument(argi++, _gpuPostNeeded);
            _queue.Execute(_collisionSPHKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, _events);

            // Read back post needed bool.
            ReadBuffer(_gpuPostNeeded, ref postNeeded, 0, 0, 1, _events);
            isPostNeeded = Convert.ToBoolean(postNeeded[0]);

            // Read back bodies as needed.
            if (isPostNeeded || alwaysReadBack || _curBufferVersion != bufferVersion)
                ReadBuffer(_gpuBodies[1], ref bodies, 0, 0, bodies.Length, _events, true);


            if (_profile)
            {
                _queue.Finish();

                ulong tot = 0;
                for (int i = 0; i < _events.Count; i++)
                {
                    var evt = _events[i];
                    var elap = evt.FinishTime - evt.StartTime;
                    tot += elap;

                    //var elapMS = (float)elap / 1000000.0f;
                    //var idxStr = i.ToString("00");
                    //Debug.WriteLine("{0} - [{3}] - Elap: {1} ns  {2} ms", idxStr, elap, elapMS, evt.Type.ToString());
                    evt.Dispose();
                }

                var ms = tot / 1000000.0f;
                _bestProfileElap = Math.Min(_bestProfileElap, ms);
                Debug.WriteLine("Elapsed ({2}): {0} ns  {1} ms", tot, ms, _bestProfileElap);
            }

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
                _queue.WriteToBuffer(bodies, inBodies, true, _events);
                _queue.Finish();

                _fixOverlapKernel.SetMemoryArgument(0, inBodies);
                _fixOverlapKernel.SetValueArgument(1, bodies.Length);
                _fixOverlapKernel.SetMemoryArgument(2, outBodies);

                _queue.Execute(_fixOverlapKernel, null, new long[] { BlockCount(bodies.Length) * _threadsPerBlock }, new long[] { _threadsPerBlock }, _events);
                _queue.Finish();

                ReadBuffer(outBodies, ref bodies, 0, 0, bodies.Length, _events);
                _queue.Finish();
            }
        }

        private void ReadMesh()
        {
            if (_mesh.Length != _meshLength)
                _mesh = new MeshCell[_meshLength];

            ReadBuffer(_gpuMeshBufs.Indexes, ref _hostMeshBufs.Indexes, 0, 0, _meshLength, _events);
            ReadBuffer(_gpuMeshBufs.NeighborBounds, ref _hostMeshBufs.NeighborBounds, 0, 0, _meshLength, _events);
            ReadBuffer(_gpuMeshBufs.BodyBounds, ref _hostMeshBufs.BodyBounds, 0, 0, _meshLength, _events);
            ReadBuffer(_gpuMeshBufs.ChildBounds, ref _hostMeshBufs.ChildBounds, 0, 0, _meshLength, _events);
            ReadBuffer(_gpuMeshBufs.CenterMass, ref _hostMeshBufs.CenterMass, 0, 0, _meshLength, _events);
            ReadBuffer(_gpuMeshBufs.SizeParentLevel, ref _hostMeshBufs.SizeParentLevel, 0, 0, _meshLength, _events);
            _queue.Finish();

            for (int i = 0; i < _meshLength; i++)
            {
                _mesh[i].IdxX = _hostMeshBufs.Indexes[i].X;
                _mesh[i].IdxY = _hostMeshBufs.Indexes[i].Y;
                _mesh[i].NeighborStartIdx = _hostMeshBufs.NeighborBounds[i].X;
                _mesh[i].NeighborCount = _hostMeshBufs.NeighborBounds[i].Y;
                _mesh[i].BodyStartIdx = _hostMeshBufs.BodyBounds[i].X;
                _mesh[i].BodyCount = _hostMeshBufs.BodyBounds[i].Y;
                _mesh[i].ChildStartIdx = _hostMeshBufs.ChildBounds[i].X;
                _mesh[i].ChildCount = _hostMeshBufs.ChildBounds[i].Y;
                _mesh[i].Mass = _hostMeshBufs.CenterMass[i].Z;
                _mesh[i].CmX = _hostMeshBufs.CenterMass[i].X;
                _mesh[i].CmY = _hostMeshBufs.CenterMass[i].Y;
                _mesh[i].Size = _hostMeshBufs.SizeParentLevel[i].X;
                _mesh[i].ParentID = _hostMeshBufs.SizeParentLevel[i].Y;
                _mesh[i].Level = _hostMeshBufs.SizeParentLevel[i].Z;
            }

            // var meshTree = BuildMeshTree();
        }

        private List<MeshCell> BuildMeshTree()
        {
            var meshTree = new List<MeshCell>();

            // Pop the root node.
            for (int i = _levelIdx[_levels]; i < _meshLength; i++)
            {
                var cell = new MeshCell();

                cell.IdxX = _hostMeshBufs.Indexes[i].X;
                cell.IdxY = _hostMeshBufs.Indexes[i].Y;
                cell.NeighborStartIdx = _hostMeshBufs.NeighborBounds[i].X;
                cell.NeighborCount = _hostMeshBufs.NeighborBounds[i].Y;
                cell.BodyStartIdx = _hostMeshBufs.BodyBounds[i].X;
                cell.BodyCount = _hostMeshBufs.BodyBounds[i].Y;
                cell.ChildStartIdx = _hostMeshBufs.ChildBounds[i].X;
                cell.ChildCount = _hostMeshBufs.ChildBounds[i].Y;
                cell.Mass = _hostMeshBufs.CenterMass[i].Z;
                cell.CmX = _hostMeshBufs.CenterMass[i].X;
                cell.CmY = _hostMeshBufs.CenterMass[i].Y;
                cell.Size = _hostMeshBufs.SizeParentLevel[i].X;
                cell.ParentID = _hostMeshBufs.SizeParentLevel[i].Y;
                cell.Level = _hostMeshBufs.SizeParentLevel[i].Z;

                meshTree.Add(cell);
            }

            // Pop all child nodes.
            var ns = ReadBuffer(_gpuMeshBufs.Neighbors);
            PopMeshTree(meshTree, _mesh, ns);

            return meshTree;
        }

        private void PopMeshTree(List<MeshCell> tree, MeshCell[] mesh, int[] neighbors)
        {
            for (int i = 0; i < tree.Count; i++)
            {
                var cell = tree[i];

                if (cell.ParentID != -1)
                {
                    cell.Parent = new List<MeshCell>() { mesh[cell.ParentID] };
                }

                if (cell.NeighborCount > 0)
                {
                    cell.Neighbors = new List<MeshCell>();
                    for (int n = cell.NeighborStartIdx; n < cell.NeighborStartIdx + cell.NeighborCount; n++)
                    {
                        cell.Neighbors.Add(mesh[neighbors[n]]);
                    }
                }

                if (cell.BodyCount > 0)
                {
                    cell.Bodies = new List<Body>();
                    for (int b = cell.BodyStartIdx; b < cell.BodyStartIdx + cell.BodyCount; b++)
                    {
                        cell.Bodies.Add(_bodies[b]);
                    }
                }

                if (cell.ChildCount > 0)
                {
                    cell.Childs = new List<MeshCell>();
                    for (int c = cell.ChildStartIdx; c < cell.ChildStartIdx + cell.ChildCount; c++)
                    {
                        cell.Childs.Add(mesh[c]);
                    }
                }

                tree[i] = cell;

                // Recurse with child nodes.
                if (tree[i].Childs != null && tree[i].Childs.Count > 0)
                    PopMeshTree(tree[i].Childs, mesh, neighbors);
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
            // Realloc level index and buffer as needed.
            if (_levelIdx.Length != _levels + 2)
            {
                // Index of mesh level locations.
                _levelIdx = new int[_levels + 2];
                _levelIdx[0] = 0; // Bottom level.

                Allocate(ref _gpuLevelIdx, _levelIdx);
            }

            // Compute a padded size.
            // The current sort kernel has particular requirements for the input size
            int padLen = ComputePaddedSize(_bodies.Length);

            // Compute Z-Order morton numbers for bodies.
            ComputeMortsGPU(padLen, cellSizeExp);

            // Sort the body morton numbers/index by the morton numbers.
            SortByMortGPU(padLen);

            // Build each level of the mesh.
            BuildMeshGPU(cellSizeExp);

            // Calc center of mass on GPU from top-most level.
            _calcCMKernel.SetMemoryArgument(0, _gpuMeshBufs.CenterMass);
            _calcCMKernel.SetMemoryArgument(1, _gpuCM);
            _calcCMKernel.SetValueArgument(2, _levelIdx[_levels]);
            _calcCMKernel.SetValueArgument(3, _meshLength);
            _queue.ExecuteTask(_calcCMKernel, _events);

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
            Allocate(ref _gpuBodyMorts[0], padLen, exact: true);

            _computeMortsKernel.SetMemoryArgument(0, _gpuBodies[1]);
            _computeMortsKernel.SetValueArgument(1, _bodies.Length);
            _computeMortsKernel.SetValueArgument(2, padLen);
            _computeMortsKernel.SetValueArgument(3, cellSizeExp);
            _computeMortsKernel.SetMemoryArgument(4, _gpuBodyMorts[0]);
            _queue.Execute(_computeMortsKernel, null, new long[] { BlockCount(padLen) * _threadsPerBlock }, new long[] { _threadsPerBlock }, _events);
        }

        //
        // Credit: https://github.com/gyatskov/radix-sort
        // https://github.com/modelflat/OCLRadixSort
        //
        private void SortByMortGPU(int padLen)
        {
            // Radix constants.
            const int _NUM_BITS_PER_RADIX = 8;
            const int _NUM_ITEMS_PER_GROUP = 16;
            const int _NUM_GROUPS = 32;
            const int _RADIX = (1 << _NUM_BITS_PER_RADIX);
            const int _DATA_SIZE = 8;
            const int _NUM_HISTOSPLIT = 512;
            const int _TOTALBITS = _DATA_SIZE << 3;
            const int _NUM_PASSES = (_TOTALBITS / _NUM_BITS_PER_RADIX);

            int numItems = _NUM_ITEMS_PER_GROUP * _NUM_GROUPS;
            int numLocalItems = _NUM_ITEMS_PER_GROUP;

            Allocate(ref _gpuBodyMorts[1], padLen, true);
            Allocate(ref _gpuHistogram, _RADIX * _NUM_GROUPS * _NUM_ITEMS_PER_GROUP, true);
            Allocate(ref _gpuGlobSum, _NUM_HISTOSPLIT, true);
            Allocate(ref _gpuGlobSumTemp, _NUM_HISTOSPLIT, true);

            for (int pass = 0; pass < _NUM_PASSES; pass++)
            {
                #region Histogram
                // Histogram
                int argi = 0;
                _histogramKernel.SetMemoryArgument(argi++, _gpuBodyMorts[0]);
                _histogramKernel.SetMemoryArgument(argi++, _gpuHistogram);
                _histogramKernel.SetValueArgument(argi++, pass);
                _histogramKernel.SetLocalArgument(argi++, SIZEOFINT * _RADIX * _NUM_ITEMS_PER_GROUP);
                _histogramKernel.SetValueArgument(argi++, padLen);
                _queue.Execute(_histogramKernel, null, new long[] { numItems }, new long[] { numLocalItems }, _events);
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
                _queue.Execute(_scanHistogramsKernel, null, new long[] { numItems }, new long[] { numLocalItems }, _events);

                // Pass 2
                numItems = _NUM_HISTOSPLIT / 2;
                numLocalItems = numItems;

                _scanHistogramsKernel.SetMemoryArgument(0, _gpuGlobSum);
                _scanHistogramsKernel.SetMemoryArgument(2, _gpuGlobSumTemp);
                _queue.Execute(_scanHistogramsKernel, null, new long[] { numItems }, new long[] { numLocalItems }, _events);
                #endregion Scan

                #region Merge
                // Merge Histograms
                numItems = _RADIX * _NUM_GROUPS * _NUM_ITEMS_PER_GROUP / 2;
                numLocalItems = numItems / _NUM_HISTOSPLIT;

                _pastehistogramsKernel.SetMemoryArgument(0, _gpuHistogram);
                _pastehistogramsKernel.SetMemoryArgument(1, _gpuGlobSum);
                _queue.Execute(_pastehistogramsKernel, null, new long[] { numItems }, new long[] { numLocalItems }, _events);

                #endregion Merge

                #region Reorder
                // Reorder
                numLocalItems = _NUM_ITEMS_PER_GROUP;
                numItems = _NUM_ITEMS_PER_GROUP * _NUM_GROUPS;

                argi = 0;
                _reorderKernel.SetMemoryArgument(argi++, _gpuBodyMorts[0]);
                _reorderKernel.SetMemoryArgument(argi++, _gpuBodyMorts[1]);
                _reorderKernel.SetMemoryArgument(argi++, _gpuHistogram);
                _reorderKernel.SetValueArgument(argi++, pass);
                _reorderKernel.SetLocalArgument(argi++, SIZEOFINT * _RADIX * _NUM_ITEMS_PER_GROUP);
                _reorderKernel.SetValueArgument(argi++, padLen);
                _queue.Execute(_reorderKernel, null, new long[] { numItems }, new long[] { numLocalItems }, _events);
                #endregion Reorder

                // Swap keys buffers.
                ComputeBuffer<long2> temp = _gpuBodyMorts[0];
                _gpuBodyMorts[0] = _gpuBodyMorts[1];
                _gpuBodyMorts[1] = temp;
            }
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

            // Allocate level counts.
            if (_levelCounts.Length != _levels + 1)
            {
                _levelCounts = new int[_levels + 1];
                Allocate(ref _gpuLevelCounts, _levelCounts);
            }

            // Allocate map and count buffers.
            Allocate(ref _gpuMap, _bodies.Length, false);
            Allocate(ref _gpuMapFlat, _bodies.Length, false);
            Allocate(ref _gpuCounts, blocks, false);

            // Allocate mesh and morts buffers to body count.
            long bufLen = AllocateMesh(_bodies.Length);
            Allocate(ref _gpuParentMorts, _bodies.Length, false);

            // Build initial map from sorted body morts.
            _cellMapKernel.SetMemoryArgument(0, _gpuBodyMorts[0]);
            _cellMapKernel.SetValueArgument(1, _bodies.Length);
            _cellMapKernel.SetMemoryArgument(2, _gpuMap);
            _cellMapKernel.SetMemoryArgument(3, _gpuCounts);
            _cellMapKernel.SetLocalArgument(4, SIZEOFINT * _threadsPerBlock);
            _cellMapKernel.SetValueArgument(5, 2); // Set step size to 2 for long2 input type.
            _cellMapKernel.SetValueArgument(6, 0);
            _cellMapKernel.SetMemoryArgument(7, _gpuLevelCounts);
            _cellMapKernel.SetValueArgument(8, _gpuBodyMorts[0].Count);
            _queue.Execute(_cellMapKernel, null, globalSize, localSize, _events);

            // Remove the gaps to compress the cell map into the beginning of the buffer.
            // This allows the map to be read properly by the mesh building kernels.
            _compressCellMapKernel.SetValueArgument(0, blocks);
            _compressCellMapKernel.SetMemoryArgument(1, _gpuMap);
            _compressCellMapKernel.SetMemoryArgument(2, _gpuMapFlat);
            _compressCellMapKernel.SetMemoryArgument(3, _gpuCounts);
            _compressCellMapKernel.SetMemoryArgument(4, _gpuLevelCounts);
            _compressCellMapKernel.SetMemoryArgument(5, _gpuLevelIdx);
            _compressCellMapKernel.SetValueArgument(6, 0);
            _queue.Execute(_compressCellMapKernel, null, globalSizeComp, localSize, _events);

            // Build the bottom mesh level, re-index bodies and compute morts for the parent level.
            int threads = 8; // Runs much faster with smaller block sizes.
            int argi = 0;
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuBodies[1]);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuBodies[0]);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuBodyMorts[0]);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuMeshBufs.Indexes);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuMeshBufs.BodyBounds);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuMeshBufs.CenterMass);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuMeshBufs.SizeParentLevel);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuLevelCounts);
            _buildBottomKernel.SetValueArgument(argi++, _bodies.Length);
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuMapFlat);
            _buildBottomKernel.SetValueArgument(argi++, cellSizeExp);
            _buildBottomKernel.SetValueArgument(argi++, (int)Math.Pow(2.0f, cellSizeExp));
            _buildBottomKernel.SetMemoryArgument(argi++, _gpuParentMorts);
            _buildBottomKernel.SetValueArgument(argi++, bufLen);
            _queue.Execute(_buildBottomKernel, null, new long[] { BlockCount(_bodies.Length, threads) * threads }, new long[] { threads }, _events);

            // Read counts from the bottom level and compute new work sizes for the parent levels.
            int[] childCounts = new int[1];
            _queue.ReadFromBuffer(_gpuLevelCounts, ref childCounts, true, 0, 0, 1, _events);

            blocks = BlockCount(childCounts[0]);
            globalSize = new long[] { blocks * _threadsPerBlock };
            globalSizeComp = new long[] { BlockCount(blocks) * _threadsPerBlock };

            // Now build the top levels of the mesh.
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
                _queue.Execute(_cellMapKernel, null, globalSize, localSize, _events);

                // Compress the cell map.
                _compressCellMapKernel.SetValueArgument(0, -1); // Same as above. Make the kernel read length from the level counts buffer.
                _compressCellMapKernel.SetMemoryArgument(1, _gpuMap);
                _compressCellMapKernel.SetMemoryArgument(2, _gpuMapFlat);
                _compressCellMapKernel.SetMemoryArgument(3, _gpuCounts);
                _compressCellMapKernel.SetMemoryArgument(4, _gpuLevelCounts);
                _compressCellMapKernel.SetMemoryArgument(5, _gpuLevelIdx);
                _compressCellMapKernel.SetValueArgument(6, level);
                _queue.Execute(_compressCellMapKernel, null, globalSizeComp, localSize, _events);

                // Build the parent level. Also computes morts for the parents parent level.
                argi = 0;
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMeshBufs.Indexes);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMeshBufs.BodyBounds);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMeshBufs.ChildBounds);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMeshBufs.CenterMass);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMeshBufs.SizeParentLevel);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuLevelCounts);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuLevelIdx);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuMapFlat);
                _buildTopKernel.SetValueArgument(argi++, (int)Math.Pow(2.0f, cellSizeExp + level));
                _buildTopKernel.SetValueArgument(argi++, level);
                _buildTopKernel.SetMemoryArgument(argi++, _gpuParentMorts);
                _buildTopKernel.SetValueArgument(argi++, bufLen);
                _queue.Execute(_buildTopKernel, null, globalSize, localSize, _events);
            }

            // Read back the level index and set the total mesh length.
            ReadBuffer(_gpuLevelIdx, ref _levelIdx, 0, 0, _levelIdx.Length, _events, true);
            _meshLength = _levelIdx[_levels + 1];

            // If the mesh buffer was too small, reallocate and rebuild it again.
            // This done because we are not reading back counts for each level and reallocating,
            // so we don't know if we have enough room until a build has completed.
            if (bufLen < _meshLength)
            {
                var newLen = AllocateMesh(_meshLength);
                Debug.WriteLine($"Mesh reallocated: {bufLen} -> {newLen}");
                Allocate(ref _gpuParentMorts, _meshLength);
                Allocate(ref _gpuMap, _meshLength);
                Allocate(ref _gpuMapFlat, _meshLength);

                BuildMeshGPU(cellSizeExp);
            }
        }

        private long AllocateMesh(int length)
        {
            if (_hostMeshBufs.Indexes.Length < length)
            {
                int newLen = (int)(length * BUF_GROW_FACTOR);
                _hostMeshBufs.Indexes = new int2[newLen];
                _hostMeshBufs.NeighborBounds = new int2[newLen];
                _hostMeshBufs.BodyBounds = new int2[newLen];
                _hostMeshBufs.ChildBounds = new int2[newLen];
                _hostMeshBufs.CenterMass = new float4[newLen];
                _hostMeshBufs.SizeParentLevel = new int4[newLen];

                long bufLen = Allocate(ref _gpuMeshBufs.Indexes, _hostMeshBufs.Indexes);
                Allocate(ref _gpuMeshBufs.NeighborBounds, _hostMeshBufs.NeighborBounds);
                Allocate(ref _gpuMeshBufs.BodyBounds, _hostMeshBufs.BodyBounds);
                Allocate(ref _gpuMeshBufs.ChildBounds, _hostMeshBufs.ChildBounds);
                Allocate(ref _gpuMeshBufs.CenterMass, _hostMeshBufs.CenterMass);
                Allocate(ref _gpuMeshBufs.SizeParentLevel, _hostMeshBufs.SizeParentLevel);

                return bufLen;
            }

            return _gpuMeshBufs.Indexes.Count;
        }

        /// <summary>
        /// Neighbor search using a binary search strategy. Faster with smaller mesh cell count.
        /// </summary>
        private void PopNeighborsBinaryGPU(int meshSize)
        {
            int topSize = meshSize - _levelIdx[1];
            int neighborLen = topSize * 9;

            Allocate(ref _gpuMeshBufs.Neighbors, neighborLen);

            int workSize = BlockCount(topSize) * _threadsPerBlock;

            int argi = 0;
            _buildNeighborsBinaryKernel.SetMemoryArgument(argi++, _gpuMeshBufs.Indexes);
            _buildNeighborsBinaryKernel.SetMemoryArgument(argi++, _gpuMeshBufs.NeighborBounds);
            _buildNeighborsBinaryKernel.SetMemoryArgument(argi++, _gpuMeshBufs.SizeParentLevel);
            _buildNeighborsBinaryKernel.SetMemoryArgument(argi++, _gpuMeshBufs.Neighbors);
            _buildNeighborsBinaryKernel.SetMemoryArgument(argi++, _gpuLevelIdx);
            _buildNeighborsBinaryKernel.SetValueArgument(argi++, topSize);
            _buildNeighborsBinaryKernel.SetValueArgument(argi++, _levelIdx[1]);
            _queue.Execute(_buildNeighborsBinaryKernel, null, new long[] { workSize }, new long[] { _threadsPerBlock }, _events);
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
            Allocate(ref _gpuMeshBufs.Neighbors, neighborLen);

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
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMeshBufs.Indexes);
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMeshBufs.SizeParentLevel);
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMeshBufs.NeighborBounds);
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMeshBufs.ChildBounds);
                _buildNeighborsMeshKernel.SetMemoryArgument(argi++, _gpuMeshBufs.Neighbors);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, _levelIdx[1]);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, _levels);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, level);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, start);
                _buildNeighborsMeshKernel.SetValueArgument(argi++, end);
                _queue.Execute(_buildNeighborsMeshKernel, null, new long[] { workSize }, new long[] { _threadsPerBlock }, _events);
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
                    _queue.CopyBuffer(buffer, newBuf, 0, 0, buffer.Count, _events);

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

        private long Allocate<T>(ref ComputeBuffer<T> buffer, T[] data) where T : struct
        {
            // Record the current flags.
            var flags = buffer.Flags;

            buffer.Dispose();
            buffer = new ComputeBuffer<T>(_context, flags, data);

            return buffer.Count;
        }

        private T[] ReadBuffer<T>(ComputeBuffer<T> buffer, bool blocking = false) where T : struct
        {
            T[] buf = new T[buffer.Count];

            if (_hasUnifiedMemory)
                blocking = true;

            _queue.ReadFromBuffer(buffer, ref buf, blocking, _events);
            if (blocking) _queue.Finish(); // This is probably redundant...

            return buf;
        }

        private T[] ReadBuffer<T>(ComputeBuffer<T> buffer, long offset, long length, bool blocking = false) where T : struct
        {
            T[] buf = new T[length - offset];

            if (_hasUnifiedMemory)
                blocking = true;

            _queue.ReadFromBuffer(buffer, ref buf, blocking, offset, 0, length - offset, _events);
            if (blocking) _queue.Finish(); // This is probably redundant...

            return buf;
        }

        private void ReadBuffer<T>(ComputeBufferBase<T> source, ref T[] dest, long sourceOffset, long destOffset, long region, ComputeEventList events, bool blocking = false) where T : struct
        {
            var sizeofT = Marshal.SizeOf<T>();
            GCHandle destinationGCHandle = GCHandle.Alloc(dest, GCHandleType.Pinned);
            IntPtr destinationOffsetPtr = Marshal.UnsafeAddrOfPinnedArrayElement(dest, (int)destOffset);

            if (_hasUnifiedMemory)
                blocking = true;

            if (_profile)
            {
                CLEventHandle[] eventHandles = ComputeTools.ExtractHandles(events, out var eventWaitListSize);
                bool eventsWritable = events != null && !events.IsReadOnly;
                CLEventHandle[] newEventHandle = eventsWritable ? new CLEventHandle[1] : null;

                Cloo.Bindings.CL12.EnqueueReadBuffer(
                   _queue.Handle,
                   source.Handle,
                   blocking,
                   new IntPtr(sourceOffset * sizeofT),
                   new IntPtr(region * sizeofT),
                   destinationOffsetPtr,
                   eventWaitListSize,
                   eventHandles,
                   newEventHandle);

                if (eventsWritable)
                    events.Add(new MyComputeEvent(newEventHandle[0], _queue));
            }
            else
            {
                Cloo.Bindings.CL12.EnqueueReadBuffer(
                   _queue.Handle,
                   source.Handle,
                   blocking,
                   new IntPtr(sourceOffset * sizeofT),
                   new IntPtr(region * sizeofT),
                   destinationOffsetPtr,
                   0,
                   null,
                   null);
            }

            destinationGCHandle.Free();
        }

        private void WriteBuffer<T>(T[] source, ComputeBufferBase<T> dest, long sourceOffset, long destOffset, long region, ComputeEventList events) where T : struct
        {
            var sizeofT = Marshal.SizeOf<T>();
            GCHandle sourceGCHandle = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr sourceOffsetPtr = Marshal.UnsafeAddrOfPinnedArrayElement(source, (int)sourceOffset);

            bool blocking = false;
            if (_hasUnifiedMemory)
                blocking = true;

            if (_profile)
            {
                CLEventHandle[] eventHandles = ComputeTools.ExtractHandles(events, out var eventWaitListSize);
                bool eventsWritable = events != null && !events.IsReadOnly;
                CLEventHandle[] newEventHandle = eventsWritable ? new CLEventHandle[1] : null;

                Cloo.Bindings.CL12.EnqueueWriteBuffer(
                    _queue.Handle,
                    dest.Handle,
                    blocking,
                    new IntPtr(destOffset * sizeofT),
                    new IntPtr(region * sizeofT),
                    sourceOffsetPtr,
                    eventWaitListSize,
                    eventHandles,
                    newEventHandle);

                if (eventsWritable)
                    events.Add(new MyComputeEvent(newEventHandle[0], _queue));
            }
            else
            {
                Cloo.Bindings.CL12.EnqueueWriteBuffer(
                   _queue.Handle,
                   dest.Handle,
                   blocking,
                   new IntPtr(destOffset * sizeofT),
                   new IntPtr(region * sizeofT),
                   sourceOffsetPtr,
                   0,
                   null,
                   null);
            }

            sourceGCHandle.Free();
        }

        public void Flush()
        {
            _curBufferVersion = int.MaxValue;

            _meshRequested.Set();
            _meshLength = 0;
            _gpuMeshBufs.Neighbors.Dispose();
            _gpuBodies[0].Dispose();
            _gpuBodies[1].Dispose();
            _gpuCM.Dispose();
            _gpuPostNeeded.Dispose();

            _gpuCounts.Dispose();
            _gpuParentMorts.Dispose();
            _gpuMap.Dispose();
            _gpuMapFlat.Dispose();
            _gpuBodyMorts[0].Dispose();
            _gpuBodyMorts[1].Dispose();
            _gpuLevelCounts.Dispose();
            _gpuLevelIdx.Dispose();
            _gpuHistogram.Dispose();
            _gpuGlobSum.Dispose();
            _gpuGlobSumTemp.Dispose();
            _gpuMeshBufs.Indexes.Dispose();
            _gpuMeshBufs.NeighborBounds.Dispose();
            _gpuMeshBufs.BodyBounds.Dispose();
            _gpuMeshBufs.ChildBounds.Dispose();
            _gpuMeshBufs.CenterMass.Dispose();
            _gpuMeshBufs.SizeParentLevel.Dispose();

            _mesh = new MeshCell[0];
            _currentFrame = 0;
            _lastMeshReadFrame = 0;
            InitBuffers();
        }

        public void Dispose()
        {
            _gpuMeshBufs.Neighbors.Dispose();
            _gpuBodies[0].Dispose();
            _gpuBodies[1].Dispose();
            _gpuCM.Dispose();
            _gpuPostNeeded.Dispose();
            _gpuCounts.Dispose();
            _gpuParentMorts.Dispose();
            _gpuMap.Dispose();
            _gpuMapFlat.Dispose();
            _gpuBodyMorts[0].Dispose();
            _gpuLevelCounts.Dispose();
            _gpuLevelIdx.Dispose();
            _gpuBodyMorts[1].Dispose();
            _gpuHistogram.Dispose();
            _gpuGlobSum.Dispose();
            _gpuGlobSumTemp.Dispose();
            _gpuMeshBufs.Indexes.Dispose();
            _gpuMeshBufs.NeighborBounds.Dispose();
            _gpuMeshBufs.BodyBounds.Dispose();
            _gpuMeshBufs.ChildBounds.Dispose();
            _gpuMeshBufs.CenterMass.Dispose();
            _gpuMeshBufs.SizeParentLevel.Dispose();



            _forceKernel.Dispose();
            _collisionSPHKernel.Dispose();
            _collisionElasticKernel.Dispose();
            _buildNeighborsMeshKernel.Dispose();
            _buildNeighborsBinaryKernel.Dispose();
            _fixOverlapKernel.Dispose();
            _buildBottomKernel.Dispose();
            _buildTopKernel.Dispose();
            _calcCMKernel.Dispose();
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