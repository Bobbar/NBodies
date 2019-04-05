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
using System.Numerics;

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
            queue = new ComputeCommandQueue(context, device, ComputeCommandQueueFlags.Profiling);

            StreamReader streamReader = new StreamReader(Environment.CurrentDirectory + "/Physics/Kernels.cl");
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
                System.IO.File.WriteAllText("build_error.txt", buildLog);
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
            collisionKernel.SetValueArgument(argi++, centerMass);
            collisionKernel.SetValueArgument(argi++, cullDistance);

            queue.Execute(collisionKernel, null, new long[] { threadBlocks * threadsPerBlock }, new long[] { threadsPerBlock }, null);
            queue.Finish();

            queue.ReadFromBuffer(_gpuInBodies, ref bodies, true, null);
            queue.Finish();

            FreeBuffers();

            _warmUp = false;
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
        /// <param name="bodies">Current field.</param>
        /// <param name="cellSizeExp">Cell size exponent. "Math.Pow(2, exponent)"</param>
        /// <param name="cellCount">Number of unique cell indexes.</param>
        /// <param name="cellStartIdx">Array containing starting indexes of each cell within the returned array.</param>
        /// <returns></returns>
        private LevelInfo CalcBodySpatials(int cellSizeExp)
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
            var cellIdx = new List<int>(_bodies.Length);
            Body[] sortBodies = new Body[_bodies.Length];

            for (int i = 0; i < spatials.Length; i++)
            {
                // Build a new sorted body array from the sorted spatial info.
                sortBodies[i] = _bodies[spatials[i].Index];

                // Update spatials index to match new sorted bodies.
                spatials[i].Index = i;

                if (val != spatials[i].Mort)
                {
                    count++;
                    val = spatials[i].Mort;
                    cellIdx.Add(i);
                }
            }

            cellIdx.Add(spatials.Length);

            // Update the original body array with the sorted one.
            _bodies = sortBodies;

            var output = new LevelInfo();
            output.Spatials = spatials;
            output.CellCount = count;
            output.CellIndex = cellIdx.ToArray();

            return output;
        }


        private LevelInfo[] CalcTopSpatials(LevelInfo bottom)
        {
            LevelInfo[] output = new LevelInfo[_levels + 1];

            output[0] = bottom;

            for (int level = 1; level <= _levels; level++)
            {
                LevelInfo current;

                if (level == 1)
                {
                    current = bottom;
                }
                else
                {
                    current = output[level - 1];
                }

                output[level] = new LevelInfo();
                output[level].Spatials = new SpatialInfo[current.CellCount];

                Parallel.ForEach(Partitioner.Create(0, current.CellCount), _parallelOptions, (range) =>
                {
                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        var spatial = current.Spatials[current.CellIndex[i]];
                        int idxX = spatial.IdxX >> 1;
                        int idxY = spatial.IdxY >> 1;
                        int morton = MortonNumber(idxX, idxY);

                        output[level].Spatials[i] = new SpatialInfo(morton, idxX, idxY, spatial.Index + i);
                    }

                });

                int count = 0;
                int val = int.MaxValue;
                var cellIdx = new List<int>(output[level].Spatials.Length);
                for (int i = 0; i < output[level].Spatials.Length; i++)
                {
                    if (val != output[level].Spatials[i].Mort)
                    {
                        count++;
                        val = output[level].Spatials[i].Mort;
                        cellIdx.Add(i);
                    }
                }

                cellIdx.Add(output[level].Spatials.Length);

                output[level].CellCount = count;
                output[level].CellIndex = cellIdx.ToArray();

            }

            return output;
        }

        /// <summary>
        /// Builds the particle mesh and mesh-neighbor index for the current field.  Also begins writing the body array to GPU...
        /// </summary>
        /// <param name="bodies">Array of bodies.</param>
        /// <param name="cellSizeExp">Cell size exponent. 2 ^ exponent = cell size.</param>
        private void BuildMesh(int cellSizeExp)
        {
            // Get spatial info for the cells about to be constructed.
            LevelInfo botLevel = CalcBodySpatials(cellSizeExp);
            var topLevels = CalcTopSpatials(botLevel);

            int totCells = 0;
            foreach (var lvl in topLevels)
            {
                totCells += lvl.CellCount;
            }

            var meshArr = new MeshCell[totCells];
            _gridInfo = new GridInfo[_levels + 1];
            object lockObj = new object();
            MinMax minMax = new MinMax();

            int cellCount = botLevel.CellCount;
            int[] cellStartIdx = botLevel.CellIndex;
            int cellSize = (int)Math.Pow(2, cellSizeExp);

            // Use the spatial info to quickly construct the first level of mesh cells in parallel.
            Parallel.ForEach(Partitioner.Create(0, cellCount), _parallelOptions, (range) =>
            {
                var mm = new MinMax();

                for (int m = range.Item1; m < range.Item2; m++)
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

            // Index to hold the starting indexes for each level.
            int[] levelIdx = new int[_levels + 1];
            levelIdx[0] = 0;

            BuildTopLevels(meshArr, ref levelIdx, topLevels, cellSizeExp, _levels);

            // Get the completed mesh and level index.
            _mesh = meshArr;
            _levelIdx = levelIdx;

            // Allocate and begin writing mesh to GPU.
            _gpuMesh = new ComputeBuffer<MeshCell>(context, ComputeMemoryFlags.ReadWrite, _mesh.Length, IntPtr.Zero);
            queue.WriteToBuffer(_mesh, _gpuMesh, false, null);

            // Populate the grid index and calulate mesh neighbor index.
            PopGridAndNeighborsGPU(_gridInfo, _mesh.Length);
        }

        private void BuildTopLevels(MeshCell[] mesh, ref int[] levelIdx, LevelInfo[] levelInfo, int cellSizeExp, int levels)
        {
            int meshOffset = 0;
            for (int level = 1; level <= levels; level++)
            {
                int cellSizeExpLevel = cellSizeExp + level;
                int cellSize = (int)Math.Pow(2, cellSizeExpLevel);
              
                meshOffset += levelInfo[level - 1].CellCount;
                levelIdx[level] = meshOffset;

                int levelOffset = 0;

                if (level > 1)
                    levelOffset = levelIdx[level - 1];

                LevelInfo current = levelInfo[level];
                int cellCount = current.CellCount;
                object lockObj = new object();
                MinMax minMax = new MinMax();

                Parallel.ForEach(Partitioner.Create(0, cellCount), _parallelOptions, (range) =>
                {
                    var mm = new MinMax();

                    for (int m = range.Item1; m < range.Item2; m++)
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
                            var child = mesh[i + levelOffset];
                            newCell.Mass += child.Mass;
                            newCell.CmX += (float)child.Mass * child.CmX;
                            newCell.CmY += (float)child.Mass * child.CmY;
                            newCell.BodyCount += child.BodyCount;
                            newCell.ChildCount++;
                            child.ParentID = newIdx;
                            mesh[i + levelOffset] = child;
                        }

                        newCell.CmX = newCell.CmX / (float)newCell.Mass;
                        newCell.CmY = newCell.CmY / (float)newCell.Mass;

                        mesh[newIdx] = newCell;
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

            _gridIndexSize = gridSize;
            _meshLen = meshSize;

            // Reallocate and resize GPU buffer as needed.
            int newCap = Allocate(ref _gpuGridIndex, nameof(_gpuGridIndex), gridSize);
            if (newCap > 0)
            {
                //clearNewGridKernel.SetMemoryArgument(0, _gpuGridIndex);
                //clearNewGridKernel.SetValueArgument(1, gridSize);

                //queue.Execute(clearNewGridKernel, null, new long[] { BlockCount(gridSize) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                //queue.Finish();

                queue.FillBuffer<int>(_gpuGridIndex, new int[1] { 0 }, 0, newCap, null);
                queue.Finish();
            }

            // Calulate total size of 1D mesh neighbor index.
            int neighborLen = meshSize * 9;
            _neighborLen = neighborLen;

            Allocate(ref _gpuMeshNeighbors, nameof(_gpuMeshNeighbors), neighborLen);

            using (var gpuGridInfo = new ComputeBuffer<GridInfo>(context, ComputeMemoryFlags.ReadOnly, gridInfo.Length, IntPtr.Zero))
            {
                // Write Grid info.
                queue.WriteToBuffer(gridInfo, gpuGridInfo, true, null);
                queue.Finish();

                // Pop grid.
                int argi = 0;
                popGridKernel.SetMemoryArgument(argi++, _gpuGridIndex);
                popGridKernel.SetMemoryArgument(argi++, gpuGridInfo);
                popGridKernel.SetMemoryArgument(argi++, _gpuMesh);
                popGridKernel.SetValueArgument(argi++, meshSize);

                queue.Execute(popGridKernel, null, new long[] { BlockCount(meshSize) * _threadsPerBlock }, new long[] { _threadsPerBlock }, null);
                queue.Finish();

                // Build neighbor index.
                argi = 0;
                buildNeighborsKernel.SetMemoryArgument(argi++, _gpuMesh);
                buildNeighborsKernel.SetValueArgument(argi++, meshSize);
                buildNeighborsKernel.SetMemoryArgument(argi++, gpuGridInfo);
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

        private int Allocate<T>(ref ComputeBuffer<T> buffer, string name, int size, bool exactSize = false) where T : struct
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

                    return newCapacity;
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

                    return size;
                }

            }

            return 0;
        }

        private T[] ReadBuffer<T>(ComputeBuffer<T> buffer) where T : struct
        {
            T[] buf = new T[buffer.Count];

            queue.ReadFromBuffer(buffer, ref buf, true, null);
            queue.Finish();

            return buf;
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