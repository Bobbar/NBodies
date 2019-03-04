//using Cudafy;
//using Cudafy.Host;
//using Cudafy.Translator;
using System;
using System.Collections.Generic;
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
        private ComputeBuffer<int> _gpuLevelIdx;

        private MeshCell[] _mesh = new MeshCell[0];
        private ComputeBuffer<MeshCell> _gpuMesh;
        private int _prevMeshLen = 0;

        private int[] _meshNeighbors = new int[0];
        private ComputeBuffer<int> _gpuMeshNeighbors;
        private int _prevNeighborLen = 0;

        private Body[] _bodies = new Body[0];
        private ComputeBuffer<Body> _gpuInBodies;
        private ComputeBuffer<Body> _gpuOutBodies;
        private int _prevBodyLen = 0;

        private int _meshChildPosition = 0;
        private bool _warmUp = true;

        private ComputeContext context;
        private ComputeCommandQueue queue;
        private ComputeKernel forceKernel;
        private ComputeKernel collisionKernel;


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
            var platform = ComputePlatform.Platforms[1];
            var device = platform.Devices[0];

            context = new ComputeContext(new[] { device }, new ComputeContextPropertyList(platform), null, IntPtr.Zero);
            queue = new ComputeCommandQueue(context, device, ComputeCommandQueueFlags.None);

            StreamReader streamReader = new StreamReader("../../Physics/Kernels.c");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            ComputeProgram program = new ComputeProgram(context, clSource);

            program.Build(null, "-cl-std=CL1.2", null, IntPtr.Zero);

            Console.WriteLine(program.GetBuildLog(device));

            forceKernel = program.CreateKernel("CalcForce");
            collisionKernel = program.CreateKernel("CalcCollisions");
        }

        private Stopwatch timer = new Stopwatch();
        private Stopwatch timer2 = new Stopwatch();

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

            // Allocate GPU memory as needed.
            if (_prevMeshLen != _mesh.Length)
            {
                if (!_warmUp)
                    _gpuMesh.Dispose();

                _gpuMesh = new ComputeBuffer<MeshCell>(context, ComputeMemoryFlags.ReadWrite, _mesh.Length, IntPtr.Zero);

                _prevMeshLen = _mesh.Length;
            }

            if (_prevNeighborLen != _meshNeighbors.Length)
            {
                if (!_warmUp)
                    _gpuMeshNeighbors.Dispose();

                _gpuMeshNeighbors = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadWrite, _meshNeighbors.Length, IntPtr.Zero);
                _prevNeighborLen = _meshNeighbors.Length;
            }

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


            _gpuLevelIdx = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadWrite, _levelIdx.Length, IntPtr.Zero);

            queue.WriteToBuffer(_levelIdx, _gpuLevelIdx, true, null);
            queue.WriteToBuffer(_mesh, _gpuMesh, true, null);
            queue.WriteToBuffer(_meshNeighbors, _gpuMeshNeighbors, true, null);
            queue.WriteToBuffer(_bodies, _gpuInBodies, true, null);

            var gridSize = new dim3(threadBlocks);
            var blockSize = new dim3(threadsPerBlock);

            int gridDims = gridSize.ToArray().Length;
            int blockDims = blockSize.ToArray().Length;
            int maxDims = Math.Max(gridDims, blockDims);

            long[] blockSizeArray = blockSize.ToFixedSizeArray(maxDims);
            long[] gridSizeArray = gridSize.ToFixedSizeArray(maxDims);
            for (int i = 0; i < maxDims; i++)
                gridSizeArray[i] *= blockSizeArray[i];


            forceKernel.SetMemoryArgument(0, _gpuInBodies);
            forceKernel.SetValueArgument(1, _bodies.Length);

            forceKernel.SetMemoryArgument(2, _gpuOutBodies);
            forceKernel.SetValueArgument(3, _bodies.Length);

            forceKernel.SetMemoryArgument(4, _gpuMesh);
            forceKernel.SetValueArgument(5, _mesh.Length);

            forceKernel.SetMemoryArgument(6, _gpuMeshNeighbors);
            forceKernel.SetValueArgument(7, _meshNeighbors.Length);

            forceKernel.SetValueArgument(8, timestep);
            forceKernel.SetValueArgument(9, _levels);

            forceKernel.SetMemoryArgument(10, _gpuLevelIdx);
            forceKernel.SetValueArgument(11, _levelIdx.Length);

            queue.Execute(forceKernel, null, gridSizeArray, blockSizeArray, null);
            queue.Finish();


            collisionKernel.SetMemoryArgument(0, _gpuOutBodies);
            collisionKernel.SetValueArgument(1, _bodies.Length);

            collisionKernel.SetMemoryArgument(2, _gpuInBodies);
            collisionKernel.SetValueArgument(3, _bodies.Length);

            collisionKernel.SetMemoryArgument(4, _gpuMesh);
            collisionKernel.SetValueArgument(5, _mesh.Length);

            collisionKernel.SetMemoryArgument(6, _gpuMeshNeighbors);
            collisionKernel.SetValueArgument(7, _meshNeighbors.Length);

            collisionKernel.SetValueArgument(8, timestep);
            collisionKernel.SetValueArgument(9, viscosity);

            queue.Execute(collisionKernel, null, gridSizeArray, blockSizeArray, null);
            queue.Finish();

            queue.ReadFromBuffer(_gpuInBodies, ref bodies, true, null);
            queue.Finish();


            if (!_warmUp)
            {
                _gpuLevelIdx.Dispose();
            }

            //   queue.Finish();
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
        private static int[] B = new int[] { 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF };
        private static int[] S = new int[] { 1, 2, 4, 8 };

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
            var options = new ParallelOptions();
            options.MaxDegreeOfParallelism = Environment.ProcessorCount;

            Parallel.For(0, _bodies.Length, options, (b) =>
            {
                int idxX = (int)_bodies[b].PosX >> cellSizeExp;
                int idxY = (int)_bodies[b].PosY >> cellSizeExp;
                int morton = MortonNumber(idxX, idxY);

                spatials[b] = new SpatialInfo(morton, idxX, idxY, b);
                mortKeys[b] = morton;
            });

            // Sort by morton number to produce a spatially sorted array.
            Array.Sort(mortKeys, spatials);

            // Compute number of unique morton numbers to determine cell count,
            // and build the start index of each cell.
            int count = 0;
            int val = 0;
            var mortIdxs = new List<int>();
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
        /// Builds the particle mesh and mesh-neighbor index for the current field.
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
            var meshList = new List<MeshCell>(cellCount);
            var meshArr = new MeshCell[cellCount];

            // Dictionary to hold mesh cell ids for fast lookups. One for each level.
            var meshDict = new Dictionary<int, int>[_levels + 1];
            meshDict[0] = new Dictionary<int, int>();

            // Use the spatial info to quickly construct the first level of mesh cells in parallel.
            var options = new ParallelOptions();
            options.MaxDegreeOfParallelism = Environment.ProcessorCount;

            Parallel.For(0, cellCount, options, (m) =>
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

                meshArr[m] = newCell;
            });

            meshList = meshArr.ToList();

            // Calculate the final center of mass for each cell and populate the mesh dictionary.
            for (int m = 0; m < meshList.Count; m++)
            {
                var cell = meshList[m];
                cell.CmX = cell.CmX / (float)cell.Mass;
                cell.CmY = cell.CmY / (float)cell.Mass;
                meshList[m] = cell;

                meshDict[0].Add(cell.Mort, m);
            }

            // Index to hold the starting indexes for each level.
            int[] levelIdx = new int[_levels + 1];
            levelIdx[0] = 0;

            _meshChildPosition = 0;

            // Build the upper levels of the mesh.
            for (int level = 1; level <= _levels; level++)
            {
                BuildNextLevel(ref meshList, ref meshDict, ref levelIdx, cellSizeExp, level);
            }

            // Get the completed mesh and level index.
            _mesh = meshList.ToArray();
            _levelIdx = levelIdx;

            // Build mesh neighbor index for all levels.
            _meshNeighbors = BuildMeshNeighborIndex(_mesh, meshDict);
        }

        private void BuildNextLevel(ref List<MeshCell> mesh, ref Dictionary<int, int>[] meshDict, ref int[] levelIdx, int cellSizeExp, int level)
        {
            cellSizeExp += level;

            meshDict[level] = new Dictionary<int, int>();

            int cellSize = (int)Math.Pow(2, cellSizeExp);

            // Current cell index.
            int cellIdx = mesh.Count;
            levelIdx[level] = cellIdx;

            for (int m = levelIdx[level - 1]; m < levelIdx[level]; m++)
            {
                var childCell = mesh[m];

                // Calculate the parent cell position from the child body position.
                // Right bit-shift to get the x/y grid indexes.
                int idxX = (int)childCell.LocX >> cellSizeExp;
                int idxY = (int)childCell.LocY >> cellSizeExp;

                // Interleave the x/y indexes to create a morton number; use this for cell UID/Hash.
                var cellUID = MortonNumber(idxX, idxY);

                // Add cell to new parent cell.
                int parentCellId;
                if (!meshDict[level].TryGetValue(cellUID, out parentCellId))
                {
                    var newCell = new MeshCell();

                    // Convert the grid index to a real location.
                    newCell.LocX = (idxX << cellSizeExp) + (cellSize * 0.5f);
                    newCell.LocY = (idxY << cellSizeExp) + (cellSize * 0.5f);
                    newCell.IdxX = idxX;
                    newCell.IdxY = idxY;
                    newCell.Size = cellSize;
                    newCell.Mass += childCell.Mass;
                    newCell.CmX += (float)childCell.Mass * childCell.CmX;
                    newCell.CmY += (float)childCell.Mass * childCell.CmY;
                    newCell.BodyCount = childCell.BodyCount;
                    newCell.ID = cellIdx;
                    newCell.Level = level;
                    newCell.ChildStartIdx = _meshChildPosition;
                    newCell.ChildCount = 1;
                    _meshChildPosition += 1;

                    meshDict[level].Add(cellUID, newCell.ID);
                    mesh.Add(newCell);

                    childCell.ParentID = cellIdx;
                    mesh[m] = childCell;

                    cellIdx++;
                }
                else
                {
                    var parentCell = mesh[parentCellId];
                    parentCell.Mass += childCell.Mass;
                    parentCell.CmX += (float)childCell.Mass * childCell.CmX;
                    parentCell.CmY += (float)childCell.Mass * childCell.CmY;
                    parentCell.BodyCount += childCell.BodyCount;
                    parentCell.ChildCount++;

                    _meshChildPosition += 1;

                    mesh[parentCellId] = parentCell;

                    childCell.ParentID = parentCellId;
                    mesh[m] = childCell;
                }
            }

            // Calculate the final center of mass for each cell.
            for (int m = levelIdx[level]; m < mesh.Count; m++)
            {
                var cell = mesh[m];
                cell.CmX = cell.CmX / (float)cell.Mass;
                cell.CmY = cell.CmY / (float)cell.Mass;
                mesh[m] = cell;
            }
        }

        /// <summary>
        /// Builds a flattened index of mesh neighbors.
        /// </summary>
        /// <param name="mesh">Particle mesh array.</param>
        /// <param name="meshDict">Mesh cell and cell UID/Hash collection.</param>
        /// <param name="cellSize">Size of mesh cells.</param>
        private int[] BuildMeshNeighborIndex(MeshCell[] mesh, Dictionary<int, int>[] meshDict)
        {
            // Collection to store the the mesh neighbor indexes.
            // Initialized with mesh length * 9 (8 neighbors per cell plus itself).
            var neighborIdxList = new List<int>(mesh.Length * 9);
            var neighborIdx = new int[mesh.Length * 9];

            var options = new ParallelOptions();
            options.MaxDegreeOfParallelism = Environment.ProcessorCount;

            Parallel.For(0, mesh.Length, options, (m) =>
            {
                // Count of neighbors found.
                int count = 0;

                for (int x = -1; x <= 1; x++)
                {
                    for (int y = -1; y <= 1; y++)
                    {
                        // Apply the current X/Y mulipliers to the mesh grid coords to get
                        // the coordinates of a neighboring cell.
                        int nX = mesh[m].IdxX + x;
                        int nY = mesh[m].IdxY + y;

                        // Convert the new coords to a cell UID/Hash and check if the cell exists.
                        var cellUID = MortonNumber(nX, nY);

                        int nCellId;
                        if (meshDict[mesh[m].Level].TryGetValue(cellUID, out nCellId))
                        {
                            neighborIdx[(m * 9) + count] = nCellId;
                            count++;
                        }
                    }
                }

                // Pad unused elements for later removal.
                for (int i = (m * 9) + count; i < (m * 9) + 9; i++)
                {
                    neighborIdx[i] = -1;
                }
            });

            // Filter unpopulated childs to build the final index.
            for (int m = 0; m < mesh.Length; m++)
            {
                int count = 0;

                for (int i = (m * 9); i < (m * 9) + 9; i++)
                {
                    if (neighborIdx[i] != -1)
                    {
                        neighborIdxList.Add(neighborIdx[i]);
                        count++;
                    }
                }

                mesh[m].NeighborStartIdx = neighborIdxList.Count - count;
                mesh[m].NeighborCount = count;
            }

            // Return the flattened mesh neighbor index.
            return neighborIdxList.ToArray();
        }

        ///// <summary>
        ///// Calculates the gravitational forces, and SPH density/pressure.
        ///// </summary>
        //[Cudafy]
        //public static void CalcForce(GThread gpThread, Body[] inBodies, Body[] outBodies, MeshCell[] inMesh, int[] meshNeighbors, float dt, int topLevel, int[] levelIdx)
        //{
        //    float GAS_K = 0.3f;
        //    float FLOAT_EPSILON = 1.192092896e-07f;

        //    // SPH variables
        //    float ksize;
        //    float ksizeSq;
        //    float factor;
        //    float diff;
        //    float fac;

        //    float totMass;
        //    float force;
        //    float distX;
        //    float distY;
        //    float dist;
        //    float distSqrt;

        //    // Get index for the current body.
        //    int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

        //    if (a > inBodies.Length - 1)
        //        return;

        //    // Copy current body and mesh cell from memory.
        //    Body outBody = inBodies[a];
        //    MeshCell bodyCell = inMesh[outBody.MeshID];
        //    MeshCell levelCell = bodyCell;
        //    MeshCell levelCellParent = inMesh[bodyCell.ParentID];

        //    // Reset forces.
        //    outBody.ForceTot = 0;
        //    outBody.ForceX = 0;
        //    outBody.ForceY = 0;
        //    outBody.Density = 0;
        //    outBody.Pressure = 0;

        //    // Calculate initial (resting) body density.
        //    ksize = 1.0f;
        //    ksizeSq = 1.0f;
        //    factor = 1.566682f;

        //    outBody.Density = (outBody.Mass * factor);

        //    for (int level = 0; level < topLevel; level++)
        //    {
        //        int start = 0;
        //        int len = 0;

        //        // Iterate parent cell neighbors.
        //        start = levelCellParent.NeighborStartIdx;
        //        len = start + levelCellParent.NeighborCount;

        //        for (int nc = start; nc < len; nc++)
        //        {
        //            int nId = meshNeighbors[nc];
        //            MeshCell nCell = inMesh[nId];

        //            // Iterate neighbor child cells.
        //            int childStartIdx = nCell.ChildStartIdx;
        //            int childLen = childStartIdx + nCell.ChildCount;

        //            for (int c = childStartIdx; c < childLen; c++)
        //            {
        //                // Make sure the current cell index is not a neighbor or this body's cell.
        //                if (c != outBody.MeshID)
        //                {
        //                    MeshCell cell = inMesh[c];

        //                    if (IsNear(levelCell, cell) == 0)
        //                    {
        //                        // Calculate the force from the cells center of mass.
        //                        distX = cell.CmX - outBody.PosX;
        //                        distY = cell.CmY - outBody.PosY;
        //                        dist = (distX * distX) + (distY * distY);

        //                        distSqrt = (float)Math.Sqrt(dist);

        //                        totMass = (float)cell.Mass * outBody.Mass;
        //                        force = totMass / dist;

        //                        outBody.ForceTot += force;
        //                        outBody.ForceX += (force * distX / distSqrt);
        //                        outBody.ForceY += (force * distY / distSqrt);
        //                    }
        //                }
        //            }
        //        }

        //        // Move up to next level.
        //        levelCell = levelCellParent;
        //        levelCellParent = inMesh[levelCellParent.ParentID];
        //    }

        //    // Iterate the top level cells.
        //    for (int top = levelIdx[topLevel]; top < inMesh.Length; top++)
        //    {
        //        MeshCell cell = inMesh[top];

        //        if (IsNear(levelCell, cell) == 0)
        //        {
        //            distX = cell.CmX - outBody.PosX;
        //            distY = cell.CmY - outBody.PosY;
        //            dist = (distX * distX) + (distY * distY);

        //            distSqrt = (float)Math.Sqrt(dist);

        //            totMass = (float)cell.Mass * outBody.Mass;
        //            force = totMass / dist;

        //            outBody.ForceTot += force;
        //            outBody.ForceX += (force * distX / distSqrt);
        //            outBody.ForceY += (force * distY / distSqrt);
        //        }
        //    }

        //    // Accumulate forces from all bodies within neighboring cells. [THIS INCLUDES THE BODY'S OWN CELL]
        //    // Read from the flattened mesh-neighbor index at the correct location.
        //    for (int n = bodyCell.NeighborStartIdx; n < bodyCell.NeighborStartIdx + bodyCell.NeighborCount; n++)
        //    {
        //        // Get the mesh cell index, then copy it from memory.
        //        int nId = meshNeighbors[n];
        //        MeshCell cell = inMesh[nId];

        //        // Iterate the bodies within the cell.
        //        // Read from the flattened mesh-body index at the correct location.
        //        int mbStart = cell.BodyStartIdx;
        //        int mbLen = cell.BodyCount + mbStart;
        //        for (int mb = mbStart; mb < mbLen; mb++)
        //        {
        //            // Save us from ourselves.
        //            if (mb != a)
        //            {
        //                Body inBody = inBodies[mb];

        //                distX = inBody.PosX - outBody.PosX;
        //                distY = inBody.PosY - outBody.PosY;
        //                dist = (distX * distX) + (distY * distY);

        //                // If this body is within collision/SPH distance.
        //                if (dist <= ksize)
        //                {
        //                    // Clamp SPH softening distance.
        //                    if (dist < FLOAT_EPSILON)
        //                    {
        //                        dist = FLOAT_EPSILON;
        //                    }

        //                    // Accumulate density.
        //                    diff = ksizeSq - dist;
        //                    fac = factor * diff * diff * diff;
        //                    outBody.Density += outBody.Mass * fac;
        //                }

        //                // Clamp gravity softening distance.
        //                if (dist < 0.04f)
        //                {
        //                    dist = 0.04f;
        //                }

        //                // Accumulate body-to-body force.
        //                distSqrt = (float)Math.Sqrt(dist);

        //                totMass = inBody.Mass * outBody.Mass;
        //                force = totMass / dist;

        //                outBody.ForceTot += force;
        //                outBody.ForceX += (force * distX / distSqrt);
        //                outBody.ForceY += (force * distY / distSqrt);
        //            }
        //        }
        //    }

        //    gpThread.SyncThreads();

        //    // Calculate pressure from density.
        //    outBody.Pressure = GAS_K * outBody.Density;

        //    if (outBody.ForceTot > outBody.Mass * 4 & outBody.Flag == 0)
        //    {
        //        outBody.InRoche = 1;
        //    }
        //    else if (outBody.ForceTot * 2 < outBody.Mass * 4)
        //    {
        //        outBody.InRoche = 0;
        //    }
        //    else if (outBody.Flag == 2 || outBody.IsExplosion == 1)
        //    {
        //        outBody.InRoche = 1;
        //    }

        //    if (outBody.Flag == 2)
        //        outBody.InRoche = 1;

        //    // Write back to memory.
        //    outBodies[a] = outBody;
        //}

        ///// <summary>
        ///// Tests the specified cell index to see if it falls within the specified range of neighbor cell indexes.
        ///// </summary>
        //[Cudafy]
        //public static int IsNear(MeshCell testCell, MeshCell neighborCell)
        //{
        //    int match = 0;

        //    for (int x = -1; x <= 1; x++)
        //    {
        //        for (int y = -1; y <= 1; y++)
        //        {
        //            if (neighborCell.IdxX == testCell.IdxX + x && neighborCell.IdxY == testCell.IdxY + y)
        //                match = 1;
        //        }
        //    }

        //    return match;
        //}

        ///// <summary>
        ///// Calculates elastic and SPH collision forces then integrates movement.
        ///// </summary>
        //[Cudafy]
        //public static void CalcCollisions(GThread gpThread, Body[] inBodies, Body[] outBodies, MeshCell[] inMesh, int[] meshNeighbors, float dt, float viscosity)
        //{
        //    float distX;
        //    float distY;
        //    float dist;
        //    float distSqrt;

        //    // Get index for the current body.
        //    int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

        //    if (a > inBodies.Length - 1)
        //        return;

        //    // Copy current body from memory.
        //    Body outBody = inBodies[a];

        //    // Copy this body's mesh cell from memory.
        //    MeshCell bodyCell = inMesh[outBody.MeshID];

        //    // Iterate neighbor cells.
        //    for (int i = bodyCell.NeighborStartIdx; i < bodyCell.NeighborStartIdx + bodyCell.NeighborCount; i++)
        //    {
        //        // Get the neighbor cell from the index.
        //        int nId = meshNeighbors[i];
        //        MeshCell cell = inMesh[nId];

        //        // Iterate the neighbor cell bodies.
        //        int mbStart = cell.BodyStartIdx;
        //        int mbLen = cell.BodyCount + mbStart;
        //        for (int mb = mbStart; mb < mbLen; mb++)
        //        {
        //            // Double tests are bad.
        //            if (mb != a)
        //            {
        //                Body inBody = inBodies[mb];

        //                distX = outBody.PosX - inBody.PosX;
        //                distY = outBody.PosY - inBody.PosY;
        //                dist = (distX * distX) + (distY * distY);

        //                // Calc the distance and check for collision.
        //                float colDist = (outBody.Size * 0.5f) + (inBody.Size * 0.5f);
        //                if (dist <= colDist * colDist)
        //                {
        //                    // We know we have a collision, so go ahead and do the expensive square root now.
        //                    distSqrt = (float)Math.Sqrt(dist);

        //                    // If both bodies are in Roche, we do SPH physics.
        //                    // Otherwise, an elastic collision and merge is done.

        //                    // SPH collision.
        //                    if (outBody.InRoche == 1 && inBody.InRoche == 1)
        //                    {
        //                        float FLOAT_EPSILON = 1.192092896e-07f;
        //                        float FLOAT_EPSILONSQRT = 3.45267E-11f;
        //                        float m_kernelSize = 1.0f;

        //                        if (dist < FLOAT_EPSILON)
        //                        {
        //                            dist = FLOAT_EPSILON;
        //                            distSqrt = FLOAT_EPSILONSQRT;
        //                        }

        //                        // Pressure force
        //                        float scalar = outBody.Mass * (outBody.Pressure + inBody.Pressure) / (2.0f * outBody.Density);
        //                        float gradFactor = -10442.157f * (m_kernelSize - distSqrt) * (m_kernelSize - distSqrt) / distSqrt;

        //                        float gradX = (distX * gradFactor);
        //                        float gradY = (distY * gradFactor);

        //                        gradX = gradX * scalar;
        //                        gradY = gradY * scalar;

        //                        outBody.ForceX -= gradX;
        //                        outBody.ForceY -= gradY;

        //                        // Viscosity force
        //                        float visc_Laplace = 14.323944f * (m_kernelSize - distSqrt);
        //                        float visc_scalar = inBody.Mass * visc_Laplace * viscosity * 1.0f / inBody.Density;

        //                        float viscVelo_diffX = inBody.VeloX - outBody.VeloX;
        //                        float viscVelo_diffY = inBody.VeloY - outBody.VeloY;

        //                        viscVelo_diffX *= visc_scalar;
        //                        viscVelo_diffY *= visc_scalar;

        //                        outBody.ForceX += viscVelo_diffX;
        //                        outBody.ForceY += viscVelo_diffY;
        //                    }
        //                    // Elastic collision.
        //                    else if (outBody.InRoche == 1 && inBody.InRoche == 0) // Out of roche bodies always consume in roche bodies.
        //                    {
        //                        outBody.Visible = 0; // Our body is merging with another body, somewhere in a far off thread.
        //                    }
        //                    else
        //                    {
        //                        // Calculate elastic collision forces.
        //                        float dotProd = distX * (inBody.VeloX - outBody.VeloX) + distY * (inBody.VeloY - outBody.VeloY);
        //                        float colScale = dotProd / dist;
        //                        float colForceX = distX * colScale;
        //                        float colForceY = distY * colScale;
        //                        float colMass = inBody.Mass / (inBody.Mass + outBody.Mass);

        //                        // If we're the bigger one, eat the other guy.
        //                        if (outBody.Mass > inBody.Mass)
        //                        {
        //                            outBody = CollideBodies(outBody, inBody, colMass, colForceX, colForceY);
        //                        }
        //                        else if (outBody.Mass < inBody.Mass) // We're smaller, so we must go away.
        //                        {
        //                            outBody.Visible = 0;
        //                        }
        //                        else if (outBody.Mass == inBody.Mass)  // If we are the same size, use a different metric.
        //                        {
        //                            // Our UID is more gooder, eat the other guy.
        //                            if (outBody.UID > inBody.UID)
        //                            {
        //                                outBody = CollideBodies(outBody, inBody, colMass, colForceX, colForceY);
        //                            }
        //                            else // Our UID is inferior, we must go away.
        //                            {
        //                                outBody.Visible = 0;
        //                            }
        //                        }
        //                    }
        //                }
        //            }
        //        }
        //    }

        //    gpThread.SyncThreads();

        //    // Integrate.
        //    outBody.VeloX += dt * outBody.ForceX / outBody.Mass;
        //    outBody.VeloY += dt * outBody.ForceY / outBody.Mass;
        //    outBody.PosX += dt * outBody.VeloX;
        //    outBody.PosY += dt * outBody.VeloY;

        //    if (outBody.Lifetime > 0.0f)
        //        outBody.Age += (dt * 4.0f);

        //    // Write back to memory.
        //    outBodies[a] = outBody;
        //}

        //[Cudafy]
        //public static Body CollideBodies(Body master, Body slave, float colMass, float forceX, float forceY)
        //{
        //    Body bodyA = master;
        //    Body bodyB = slave;

        //    bodyA.VeloX += colMass * forceX;
        //    bodyA.VeloY += colMass * forceY;

        //    if (bodyA.Flag != 1)
        //    {
        //        float a1 = (float)Math.PI * (float)(Math.Pow(bodyA.Size * 0.5f, 2));
        //        float a2 = (float)Math.PI * (float)(Math.Pow(bodyB.Size * 0.5f, 2));
        //        float a = a1 + a2;
        //        bodyA.Size = (float)Math.Sqrt((float)(a / Math.PI)) * 2;
        //    }

        //    bodyA.Mass += bodyB.Mass;

        //    return bodyA;
        //}
    }

    /// <summary>
    /// Cudafy equivalent of Cuda dim3.
    /// </summary>
    public class dim3
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="dim3"/> class. Y and z will be 1.
        /// </summary>
        /// <param name="x">The x value.</param>
        public dim3(int x)
        {
            this.x = x;
            this.y = 1;
            this.z = 1;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="dim3"/> class. Z will be 1.
        /// </summary>
        /// <param name="x">The x value.</param>
        /// <param name="y">The y value.</param>
        public dim3(int x, int y)
        {
            this.x = x;
            this.y = y;
            this.z = 1;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="dim3"/> class.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        public dim3(long[] dimensions)
        {
            int len = dimensions.Length;
            if (len > 0)
                x = (int)dimensions[0];
            if (len > 1)
                y = (int)dimensions[1];
            if (len > 2)
                z = (int)dimensions[2];
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="dim3"/> class.
        /// </summary>
        /// <param name="x">The x value.</param>
        /// <param name="y">The y value.</param>
        /// <param name="z">The z value.</param>
        public dim3(long x, long y, long z)
        {
            this.x = (int)x;
            this.y = (int)y;
            this.z = (int)z;
        }

        /// <summary>
        /// Gets the x.
        /// </summary>
        public int x { get; private set; }
        /// <summary>
        /// Gets the y.
        /// </summary>
        public int y { get; private set; }
        /// <summary>
        /// Gets the z.
        /// </summary>
        public int z { get; private set; }

        /// <summary>
        /// Helper method to transform into an array of dimension sizes.
        /// </summary>
        /// <returns></returns>
        public long[] ToArray()
        {
            int dims = 1;
            if (z > 1)
                dims = 3;
            else if (y > 1)
                dims = 2;
            long[] array = new long[dims];
            array[0] = x;
            if (dims > 1)
                array[1] = y;
            if (dims > 2)
                array[2] = z;
            return array;
        }

        public long[] ToFixedSizeArray(int size)
        {
            if (size < 1 || size > 3)
                throw new ArgumentOutOfRangeException("size");
            long[] array = new long[size];
            array[0] = x;
            if (size > 1)
                array[1] = y;
            if (size > 2)
                array[2] = z;
            return array;
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="System.Int32"/> to <see cref="Cudafy.dim3"/>.
        /// </summary>
        /// <param name="dimX">The dim X.</param>
        /// <returns>
        /// The result of the conversion.
        /// </returns>
        public static implicit operator dim3(int dimX) { return new dim3(dimX); }
    }

}