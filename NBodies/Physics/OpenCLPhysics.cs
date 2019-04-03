using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Cloo;
using Cloo.Bindings;
using Cloo.Extensions;
using System.IO;
using NBodies.Extensions;
using System.Runtime.InteropServices;

namespace NBodies.Physics
{
    public class OpenCLPhysics : IPhysicsCalc
    {
        private int _gpuIndex = 2;
        private int _levels = 4;
        private static int _threadsPerBlock = 256;

        private int[] _levelIdx = new int[0];

        private MeshCell[] _mesh = new MeshCell[0];

        private Body[] _bodies = new Body[0];
        private int _prevBodyLen = 0;

        private GridInfo[] _gridInfo = new GridInfo[0];

        private bool _warmUp = true;

        private int _gridIndexSize = 0;
        private int _meshLen = 0;
        private int _neighborLen = 0;

        private ComputeContext context;
        private ComputeCommandQueue queue;

        private ComputeKernel forceKernel;
        private ComputeKernel collisionKernel;
        private ComputeKernel popGridKernel;
        private ComputeKernel clearGridKernel;
        private ComputeKernel clearNewGridKernel;
        private ComputeKernel buildNeighborsKernel;

        private ComputeBuffer<int> _gpuLevelIdx;
        private ComputeBuffer<MeshCell> _gpuMesh;
        private ComputeBuffer<int> _gpuMeshNeighbors;
        private ComputeBuffer<Body> _gpuInBodies;
        private ComputeBuffer<Body> _gpuOutBodies;
        private ComputeBuffer<int> _gpuGridIndex;

        private static Dictionary<string, BufferDims> _bufferInfo = new Dictionary<string, BufferDims>();

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

        public OpenCLPhysics(int gpuIdx)
        {
            _gpuIndex = gpuIdx;
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

            context = new ComputeContext(new[] { device }, new ComputeContextPropertyList(platform), null, IntPtr.Zero);
            queue = new ComputeCommandQueue(context, device, ComputeCommandQueueFlags.None);

            StreamReader streamReader = new StreamReader("../../Physics/Kernels.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            ComputeProgram program = new ComputeProgram(context, clSource);

            try
            {
                program.Build(null, "-cl-std=CL1.2", null, IntPtr.Zero);
            }
            catch (BuildProgramFailureComputeException ex)
            {
                string buildLog = program.GetBuildLog(device);
                Console.WriteLine(buildLog);
                throw;
            }

            Console.WriteLine(program.GetBuildLog(device));

            forceKernel = program.CreateKernel("CalcForce");
            collisionKernel = program.CreateKernel("CalcCollisions");
            popGridKernel = program.CreateKernel("PopGrid");
            clearNewGridKernel = program.CreateKernel("ClearNewGrid");
            buildNeighborsKernel = program.CreateKernel("BuildNeighbors");
            clearGridKernel = program.CreateKernel("ClearGrid");

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
            if (!_warmUp)
            {
                FreeBuffers();
            }

            _gpuGridIndex.Dispose();
            _gpuMeshNeighbors.Dispose();
            _bufferInfo.Clear();

            InitBuffers();

            _warmUp = true;
        }

        private void InitBuffers()
        {

            _gpuGridIndex = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadWrite, 1, IntPtr.Zero);
            Allocate(ref _gpuGridIndex, nameof(_gpuGridIndex), 0);

            _gpuMeshNeighbors = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadWrite, 1, IntPtr.Zero);
            Allocate(ref _gpuMeshNeighbors, nameof(_gpuMeshNeighbors), 0);

        }

        private void FreeBuffers()
        {
            _gpuMesh.Dispose();
            _gpuLevelIdx.Dispose();

        }

        public void CalcMovement(ref Body[] bodies, float timestep, int cellSizeExp, int meshLevels, int threadsPerBlock)
        {
            _bodies = bodies;
            _threadsPerBlock = threadsPerBlock;
            _levels = meshLevels;
            float viscosity = 10.0f; // Viscosity for SPH particles in the collisions kernel.
            int threadBlocks = 0;

            // Calc number of thread blocks to fit the dataset.
            threadBlocks = BlockCount(_bodies.Length);

            // Build the particle mesh, mesh index, mesh child index and mesh neighbors index.
            BuildMesh(cellSizeExp);

            _gpuLevelIdx = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadOnly, _levelIdx.Length, IntPtr.Zero);

            queue.WriteToBuffer(_levelIdx, _gpuLevelIdx, true, null);
            queue.Finish();

            int argi = 0;
            forceKernel.SetMemoryArgument(argi++, _gpuInBodies);
            forceKernel.SetValueArgument(argi++, _bodies.Length);
            forceKernel.SetMemoryArgument(argi++, _gpuOutBodies);
            forceKernel.SetMemoryArgument(argi++, _gpuMesh);
            forceKernel.SetValueArgument(argi++, _mesh.Length);
            forceKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            forceKernel.SetValueArgument(argi++, timestep);
            forceKernel.SetValueArgument(argi++, _levels);
            forceKernel.SetMemoryArgument(argi++, _gpuLevelIdx);
            forceKernel.SetValueArgument(argi++, _levelIdx.Length);

            queue.Execute(forceKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);
            queue.Finish();

            argi = 0;
            collisionKernel.SetMemoryArgument(argi++, _gpuOutBodies);
            collisionKernel.SetValueArgument(argi++, _bodies.Length);
            collisionKernel.SetMemoryArgument(argi++, _gpuInBodies);
            collisionKernel.SetMemoryArgument(argi++, _gpuMesh);
            collisionKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);
            collisionKernel.SetValueArgument(argi++, timestep);
            collisionKernel.SetValueArgument(argi++, viscosity);

            queue.Execute(collisionKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);
            queue.Finish();

            queue.ReadFromBuffer(_gpuInBodies, ref bodies, true, null);
            queue.Finish();


            //if (!_warmUp)
            //{
            //    _gpuLevelIdx.Dispose();
            //}

            FreeBuffers();

            _warmUp = false;
        }

        private void WriteBodiesToGPU()
        {
            if (_prevBodyLen != _bodies.Length)
            {
                if (!_warmUp)
                {
                    _gpuInBodies.Dispose();
                    _gpuOutBodies.Dispose();
                }

                _gpuInBodies = new ComputeBuffer<Body>(context, ComputeMemoryFlags.ReadWrite, _bodies.Length, IntPtr.Zero);
                _gpuOutBodies = new ComputeBuffer<Body>(context, ComputeMemoryFlags.ReadWrite, _bodies.Length, IntPtr.Zero);
                _prevBodyLen = _bodies.Length;
            }

            queue.WriteToBuffer(_bodies, _gpuInBodies, false, null);
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

            //int columns = minMax.MaxX - (minMax.MinX - 1);
            //int rows = minMax.MaxY - (minMax.MinY - 1);

            int idxOff = 0;
            int size = ((columns + 1) * (rows + 1));

            if (minMax.MinX < -40000 || minMax.MinY < -40000)
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
        /// <param name="bodies">Current field.</param>
        /// <param name="cellSizeExp">Cell size exponent. "Math.Pow(2, exponent)"</param>
        /// <param name="cellCount">Number of unique cell indexes.</param>
        /// <param name="cellStartIdx">Array containing starting indexes of each cell within the returned array.</param>
        /// <returns></returns>
        private SpatialInfo[] CalcBodySpatials(int cellSizeExp, out int cellCount, out int[] cellStartIdx)
        {
            var spatials = new SpatialInfo[_bodies.Length];

            // Array of morton numbers used for sorting.
            // Using a key array when sorting is much faster than sorting an array of objects by a field.
            int[] mortKeys = new int[_bodies.Length];

            // Compute the spatial info in parallel.
            Parallel.ForEach(Partitioner.Create(0, _bodies.Length), _parallelOptions, (range) =>
            {
                for (int b = range.Item1; b < range.Item2; b++)
                {
                    int idxX = (int)_bodies[b].PosX >> cellSizeExp;
                    int idxY = (int)_bodies[b].PosY >> cellSizeExp;
                    int morton = MortonNumber(idxX, idxY);

                    spatials[b] = new SpatialInfo(morton, idxX, idxY, b);
                    mortKeys[b] = morton;
                }

            });

            // Sort by morton number to produce a spatially sorted array.
            Array.Sort(mortKeys, spatials);

            // Compute number of unique morton numbers to determine cell count,
            // and build the start index of each cell.
            int count = 0;
            int val = 0;
            var mortIdxs = new List<int>(_bodies.Length);
            Body[] sortBodies = new Body[_bodies.Length];

            for (int i = 0; i < spatials.Length; i++)
            {
                // Build a new sorted body array from the sorted spatial info.
                sortBodies[i] = _bodies[spatials[i].Index];

                if (val != spatials[i].Mort)
                {
                    count++;
                    val = spatials[i].Mort;
                    mortIdxs.Add(i);
                }
            }

            mortIdxs.Add(spatials.Length);

            // Update the original body array with the sorted one.
            _bodies = sortBodies;

            // Output count and start index.
            cellCount = count;
            cellStartIdx = mortIdxs.ToArray();

            return spatials;
        }

        /// <summary>
        /// Builds the particle mesh and mesh-neighbor index for the current field.  Also begins writing the body array to GPU...
        /// </summary>
        /// <param name="bodies">Array of bodies.</param>
        /// <param name="cellSizeExp">Cell size exponent. 2 ^ exponent = cell size.</param>
        private void BuildMesh(int cellSizeExp)
        {
            int cellSize = (int)Math.Pow(2, cellSizeExp);

            int cellCount;
            int[] cellStartIdx;

            // Get spatail info for the cells about to be constructed.
            SpatialInfo[] spatialDat = CalcBodySpatials(cellSizeExp, out cellCount, out cellStartIdx);

            // List to hold all new mesh cells.
            var meshList = new List<MeshCell>(_bodies.Length);
            var meshArr = new MeshCell[cellCount];

            _gridInfo = new GridInfo[_levels + 1];

            object lockObj = new object();
            MinMax minMax = new MinMax();

            // Use the spatial info to quickly construct the first level of mesh cells in parallel.
            Parallel.ForEach(Partitioner.Create(0, cellCount), _parallelOptions, (range) =>
            {
                var mm = new MinMax();

                for (int m = range.Item1; m < range.Item2; m++)
                {
                    // Get the spatial info from the first cell index; there may only be one cell.
                    var spatial = spatialDat[cellStartIdx[m]];

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

                    meshArr[m] = newCell;
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

            meshList = new List<MeshCell>(meshArr);

            // Set the mesh list capacity to slightly larger than the previous size.
            // This will reduce large reallocations from the list resizing.
            if (_mesh.Length > 0 & (_mesh.Length * 1.5) > meshList.Capacity)
                meshList.Capacity = (int)(_mesh.Length * 1.5);

            // Index to hold the starting indexes for each level.
            int[] levelIdx = new int[_levels + 1];
            levelIdx[0] = 0;

            // Build the upper levels of the mesh.
            BuildTopLevels(ref meshList, ref levelIdx, cellSizeExp, _levels);

            // Get the completed mesh and level index.
            _mesh = meshList.ToArray();
            _levelIdx = levelIdx;

            _gpuMesh = new ComputeBuffer<MeshCell>(context, ComputeMemoryFlags.ReadWrite, _mesh.Length, IntPtr.Zero);
            queue.WriteToBuffer(_mesh, _gpuMesh, false, null);

            PopGridGPU(_gridInfo, _mesh.Length);
        }

        private void BuildTopLevels(ref List<MeshCell> mesh, ref int[] levelIdx, int cellSizeExp, int levels)
        {
            int meshChildIndexPosition = 0;

            for (int level = 1; level <= levels; level++)
            {
                var minMax = new MinMax();
                int cellSizeExpLevel = cellSizeExp + level;
                int cellSize = (int)Math.Pow(2, cellSizeExpLevel);

                // Current cell index.
                int cellIdx = mesh.Count;
                levelIdx[level] = cellIdx;

                int prevUID = int.MinValue;
                MeshCell newCell = new MeshCell();

                for (int m = levelIdx[level - 1]; m < levelIdx[level]; m++)
                {
                    var childCell = mesh[m];

                    // Calculate the parent cell position from the child body position.
                    // Right bit-shift to get the x/y grid indexes.
                    int idxX = (int)childCell.LocX >> cellSizeExpLevel;
                    int idxY = (int)childCell.LocY >> cellSizeExpLevel;

                    minMax.Update(idxX, idxY);

                    // Interleave the x/y indexes to create a morton number; use this for cell UID/Hash.
                    var cellUID = MortonNumber(idxX, idxY);

                    // If the UID doesn't match the previous, we have a new cell.
                    if (prevUID != cellUID)
                    {
                        prevUID = cellUID;

                        // Set the previous completed parent cell center of mass.
                        if (newCell.ID != 0)
                        {
                            newCell.CmX = newCell.CmX / (float)newCell.Mass;
                            newCell.CmY = newCell.CmY / (float)newCell.Mass;
                            mesh[newCell.ID] = newCell;
                        }

                        // Create a new parent cell.
                        newCell = new MeshCell();

                        // Add initial values.
                        // Convert the grid index to a real location.
                        newCell.LocX = (idxX << cellSizeExpLevel) + (cellSize * 0.5f);
                        newCell.LocY = (idxY << cellSizeExpLevel) + (cellSize * 0.5f);
                        newCell.IdxX = idxX;
                        newCell.IdxY = idxY;
                        newCell.Size = cellSize;
                        newCell.Mass += childCell.Mass;
                        newCell.CmX += (float)childCell.Mass * childCell.CmX;
                        newCell.CmY += (float)childCell.Mass * childCell.CmY;
                        newCell.BodyCount = childCell.BodyCount;
                        newCell.ID = cellIdx;
                        newCell.Level = level;
                        newCell.ChildStartIdx = meshChildIndexPosition;
                        newCell.ChildCount = 1;

                        // Increment the child index position.
                        meshChildIndexPosition++;

                        mesh.Add(newCell);

                        childCell.ParentID = cellIdx;
                        mesh[m] = childCell;

                        cellIdx++;
                    }
                    else // If UID matches previous, add a child to the parent cell.
                    {
                        newCell.Mass += childCell.Mass;
                        newCell.CmX += (float)childCell.Mass * childCell.CmX;
                        newCell.CmY += (float)childCell.Mass * childCell.CmY;
                        newCell.BodyCount += childCell.BodyCount;
                        newCell.ChildCount++;

                        meshChildIndexPosition++;

                        mesh[newCell.ID] = newCell;

                        childCell.ParentID = newCell.ID;
                        mesh[m] = childCell;
                    }
                }

                // Calc CM for last cell.
                var lastCell = mesh.Last();
                lastCell.CmX = lastCell.CmX / (float)lastCell.Mass;
                lastCell.CmY = lastCell.CmY / (float)lastCell.Mass;
                mesh[lastCell.ID] = lastCell;

                AddGridDims(minMax, level);
            }
        }

        private void PopGridGPU(GridInfo[] gridInfo, int meshSize)
        {
            int gridSize = 0;
            foreach (var g in gridInfo)
            {
                gridSize += g.Size;
            }

            _gridIndexSize = gridSize;
            _meshLen = meshSize;

            bool resized = Allocate(ref _gpuGridIndex, nameof(_gpuGridIndex), gridSize);
            if (resized)
            {
                clearNewGridKernel.SetMemoryArgument(0, _gpuGridIndex);
                clearNewGridKernel.SetValueArgument(1, gridSize);

                queue.Execute(clearNewGridKernel, null, new long[] { BlockCount(gridSize) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                queue.Finish();

                //queue.FillBuffer<int>(_gpuGridIndex, new int[1] { 0 }, 0, gridSize, null);
                //queue.Finish();
            }

            int neighborLen = meshSize * 9;
            _neighborLen = neighborLen;

            Allocate(ref _gpuMeshNeighbors, nameof(_gpuMeshNeighbors), neighborLen);

            using (var _gpuGridInfo = new ComputeBuffer<GridInfo>(context, ComputeMemoryFlags.ReadOnly, gridInfo.Length, IntPtr.Zero))
            {
                // Write Grid info.
                queue.WriteToBuffer(gridInfo, _gpuGridInfo, true, null);
                queue.Finish();

                // Pop grid.
                int argi = 0;
                popGridKernel.SetMemoryArgument(argi++, _gpuGridIndex);
                popGridKernel.SetMemoryArgument(argi++, _gpuGridInfo);
                popGridKernel.SetMemoryArgument(argi++, _gpuMesh);
                popGridKernel.SetValueArgument(argi++, meshSize);

                queue.Execute(popGridKernel, null, new long[] { BlockCount(meshSize) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                queue.Finish();

                // Build neighbor index.
                argi = 0;
                buildNeighborsKernel.SetMemoryArgument(argi++, _gpuMesh);
                buildNeighborsKernel.SetValueArgument(argi++, meshSize);
                buildNeighborsKernel.SetMemoryArgument(argi++, _gpuGridInfo);
                buildNeighborsKernel.SetMemoryArgument(argi++, _gpuGridIndex);
                buildNeighborsKernel.SetValueArgument(argi++, _gridIndexSize);
                buildNeighborsKernel.SetMemoryArgument(argi++, _gpuMeshNeighbors);

                int blocks = BlockCount(meshSize);
                queue.Execute(buildNeighborsKernel, null, new long[] { blocks * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                queue.Finish();


                // We're done with the grid index, so undo what we added to clear it for the next frame.
                clearGridKernel.SetMemoryArgument(0, _gpuGridIndex);
                clearGridKernel.SetMemoryArgument(1, _gpuMesh);
                clearGridKernel.SetValueArgument(2, meshSize);

                blocks = BlockCount(meshSize);
                queue.Execute(clearGridKernel, null, new long[] { blocks * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                queue.Finish();
            }

        }

        private bool Allocate<T>(ref ComputeBuffer<T> buffer, string name, int size, bool exactSize = false) where T : struct
        {
            if (!_bufferInfo.ContainsKey(name))
            {
                _bufferInfo.Add(name, new BufferDims(name, size, size, exactSize));
            }

            var dims = _bufferInfo[name];
            var flags = buffer.Flags;

            if (!dims.ExactSize)
            {
                //if (dims.Capacity < size || size < (dims.Size * dims.ShrinkFactor))
                if (dims.Capacity < size)
                // if (dims.Capacity < size || (size * dims.ShrinkFactor) < dims.Size)
                {
                    //if (!_warmUp)
                    buffer.Dispose();

                    int newCapacity = (int)(size * dims.GrowFactor);

                    dims.Capacity = newCapacity;
                    dims.Size = size;

                    buffer = new ComputeBuffer<T>(context, flags, newCapacity, IntPtr.Zero);

                    Console.WriteLine($@"[{dims.Name}]  {dims.Capacity}");

                    return true;
                }
            }
            else
            {
                if (dims.Size != size)
                {

                    //  if (!_warmUp)
                    buffer.Dispose();

                    dims.Capacity = size;
                    dims.Size = size;

                    buffer = new ComputeBuffer<T>(context, flags, size, IntPtr.Zero);

                    //  Console.WriteLine($@"[{dims.Name}]  {dims.Capacity}");

                    return true;
                }

            }

            return false;
        }

        private T[] ReadBuffer<T>(ComputeBuffer<T> buffer) where T : struct
        {
            T[] buf = new T[buffer.Count];

            queue.ReadFromBuffer(buffer, ref buf, true, null);
            queue.Finish();

            return buf;
        }

    }
}