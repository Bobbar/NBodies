using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using NBodies.Rendering;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public class CUDAFloat : IPhysicsCalc
    {
        private int gpuIndex = 2;
        private static int _threadsPerBlock = 256;
        private GPGPU gpu;
        private MeshCell[] _mesh = new MeshCell[0];
        private int[] _levelIdx = new int[0];
        private int _levels = 4;//3;
        private int[] _meshBodies = new int[0];
        private int[] _meshNeighbors = new int[0];
        private int[] _meshChilds = new int[0];

        private MeshCell[] gpuMesh = new MeshCell[0];
        private int prevMeshLen = 0;

        private int[] gpuMeshBodies = new int[0];
        private int prevMeshBodLen = 0;

        private int[] gpuMeshNeighbors = new int[0];
        private int prevMeshNLen = 0;

        private int[] gpuMeshChilds = new int[0];
        private int prevMeshChildLen = 0;

        private Body[] gpuInBodies = new Body[0];
        private Body[] gpuOutBodies = new Body[0];
        private int prevBodyLen = 0;

        private bool warmUp = true;

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

        public CUDAFloat(int gpuIdx)
        {
            gpuIndex = gpuIdx;
        }

        public CUDAFloat(int gpuIdx, int threadsperblock)
        {
            if (gpuIdx != -1)
                gpuIndex = gpuIdx;

            if (threadsperblock != -1)
                _threadsPerBlock = threadsperblock;
        }

        public void Init()
        {
            var cudaModule = CudafyModule.TryDeserialize();

            if (cudaModule == null || !cudaModule.TryVerifyChecksums(ePlatform.x64, eArchitecture.OpenCL12))
            {
                CudafyTranslator.Language = eLanguage.OpenCL;
                cudaModule = CudafyTranslator.Cudafy(ePlatform.x64, eArchitecture.OpenCL12, new Type[] { typeof(Body), typeof(MeshCell), typeof(CUDAFloat) });
                cudaModule.Serialize();
            }

            //Add missing 'struct' strings to generated code.
            cudaModule.SourceCode = FixCode(cudaModule.SourceCode, nameof(Body), nameof(MeshCell));
            cudaModule.SourceCode = cudaModule.SourceCode.Replace("sqrt((double", "half_sqrt((float");
            cudaModule.Serialize();

            gpu = CudafyHost.GetDevice(eGPUType.OpenCL, gpuIndex);
            gpu.LoadModule(cudaModule);

            var props = gpu.GetDeviceProperties();
            Console.WriteLine(props.ToString());
        }

        /// <summary>
        /// Fixes missing 'struct' strings for each function and variable declaration.
        ///
        /// Cudafy doesn't seem to support structs correctly within functions and local variables.
        /// </summary>
        private string FixCode(string code, params string[] targets)
        {
            var rgx = new Regex("[^a-zA-Z0-9 -]");
            string newcode = string.Copy(code);

            foreach (string target in targets)
            {
                bool missingDec = true;
                int position = 0;

                // Body structs
                while (missingDec)
                {
                    // Search for target string.
                    int idx = newcode.IndexOf(target, position);

                    // Stop if no match found.
                    if (idx == -1)
                    {
                        missingDec = false;
                        continue;
                    }

                    // Move the position past the current match.
                    position = idx + target.Length;

                    // Check both sides of the located string to make sure it's a match.
                    string check = newcode.Substring(idx - 1, target.Length + 2);
                    check = rgx.Replace(check, "").Trim(); // Remove non-alpha and spaces.

                    if (check == target)
                    {
                        // Make sure 'struct' isn't already present.
                        string sub = newcode.Substring(idx - 7, 7);

                        if (!sub.Contains("struct"))
                        {
                            // Add 'struct' before the target string.
                            newcode = newcode.Insert(idx, "struct ");
                        }
                    }
                }
            }

            return newcode;
        }

        private Stopwatch timer = new Stopwatch();
        private Stopwatch timer2 = new Stopwatch();

        public void CalcMovement(ref Body[] bodies, float timestep, int cellSizeExp, int meshLevels, int threadsPerBlock)
        {
            _threadsPerBlock = threadsPerBlock;
            _levels = meshLevels;
            float viscosity = 10.0f; // Viscosity for SPH particles in the collisions kernel.
            int threadBlocks = 0;

            // Calc number of thread blocks to fit the dataset.
            threadBlocks = BlockCount(bodies.Length);

            //  timer.Restart();

            // Build the particle mesh, mesh index, and mesh neighbors index.
            BuildMesh(ref bodies, cellSizeExp);

            //  Console.WriteLine($@"Mesh({_mesh.Length}): {timer.Elapsed.Milliseconds}");

            // Allocate GPU memory as needed.
            if (prevMeshLen != _mesh.Length)
            {
                if (!warmUp)
                    gpu.Free(gpuMesh);

                gpuMesh = gpu.Allocate(_mesh);
                prevMeshLen = _mesh.Length;
            }

            if (prevMeshBodLen != _meshBodies.Length)
            {
                if (!warmUp)
                    gpu.Free(gpuMeshBodies);

                gpuMeshBodies = gpu.Allocate(_meshBodies);
                prevMeshBodLen = _meshBodies.Length;
            }

            if (prevMeshNLen != _meshNeighbors.Length)
            {
                if (!warmUp)
                    gpu.Free(gpuMeshNeighbors);

                gpuMeshNeighbors = gpu.Allocate(_meshNeighbors);
                prevMeshNLen = _meshNeighbors.Length;
            }

            if (prevMeshChildLen != _meshChilds.Length)
            {
                if (!warmUp)
                    gpu.Free(gpuMeshChilds);

                gpuMeshChilds = gpu.Allocate(_meshChilds);
                prevMeshChildLen = _meshChilds.Length;
            }

            if (prevBodyLen != bodies.Length)
            {
                if (!warmUp)
                {
                    gpu.Free(gpuInBodies);
                    gpu.Free(gpuOutBodies);
                }

                gpuInBodies = gpu.Allocate(bodies);
                gpuOutBodies = gpu.Allocate(bodies);
                prevBodyLen = bodies.Length;
            }

            int[] gpuLevelIdx = gpu.Allocate(_levelIdx);

            // Copy host arrays to GPU device.
            gpu.CopyToDevice(_levelIdx, gpuLevelIdx);
            gpu.CopyToDevice(_mesh, gpuMesh);
            gpu.CopyToDevice(_meshBodies, gpuMeshBodies);
            gpu.CopyToDevice(_meshNeighbors, gpuMeshNeighbors);
            gpu.CopyToDevice(_meshChilds, gpuMeshChilds);

            gpu.CopyToDevice(bodies, gpuInBodies);

            // Launch kernels with specified integration type.
            if (MainLoop.LeapFrog)
            {
                for (int drift = 1; drift > -1; drift--)
                {
                    //gpu.Launch(threadBlocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, gpuMesh, gpuMeshBodies, gpuMeshNeighbors, timestep);
                    //gpu.Launch(threadBlocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, gpuMesh, gpuMeshBodies, gpuMeshNeighbors, timestep, viscosity, drift);
                }
            }
            else
            {
                //  timer.Restart();

                gpu.Launch(threadBlocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, gpuMesh, gpuMeshBodies, gpuMeshNeighbors, gpuMeshChilds, timestep, _levels, gpuLevelIdx);
                //gpu.Launch(threadBlocks, threadsPerBlock).CalcForceOld(gpuInBodies, gpuOutBodies, gpuMesh, gpuMeshBodies, gpuMeshNeighbors, timestep, gpuLevelIdx);

                //    Console.WriteLine($@"Force Kernel: {timer.Elapsed.Milliseconds}");

                gpu.Launch(threadBlocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, gpuMesh, gpuMeshBodies, gpuMeshNeighbors, timestep, viscosity, 3);
            }

            // Copy updated bodies back to host and free memory.
            gpu.CopyFromDevice(gpuInBodies, bodies);

            if (!warmUp)
                gpu.Free(gpuLevelIdx);

            warmUp = false;
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
        /// Builds the particle mesh, mesh-body index and mesh-neighbor index for the current field.
        /// </summary>
        /// <param name="bodies">Array of bodies.</param>
        /// <param name="cellSizeExp">Cell size exponent. 2 ^ exponent = cell size.</param>
        private void BuildMesh(ref Body[] bodies, int cellSizeExp)
        {
            int cellSize = (int)Math.Pow(2, cellSizeExp);
            // Dictionary to hold the current mesh cells for fast lookups.
            var meshDict = new Dictionary<int, MeshCell>[_levels + 1];//(_mesh.Length);
            meshDict[0] = new Dictionary<int, MeshCell>();
            // 2D Collection to hold the indexes of bodies contained in each cell.
            var meshBods = new List<List<int>>();
            // Current cell index.
            int cellIdx = 0;

            var childIdx = new List<List<int>>();

            for (int b = 0; b < bodies.Length; b++)
            {
                var body = bodies[b];

                // Calculate the cell position from the current body position.
                // Right bit-shift to get the x/y grid indexes.

                int idxX = (int)body.LocX >> cellSizeExp;
                int idxY = (int)body.LocY >> cellSizeExp;

                // Interleave the x/y indexes to create a morton number; use this for cell UID/Hash.
                var cellUID = MortonNumber(idxX, idxY);
                MeshCell nCell;
                // Add body to new cell.
                if (!meshDict[0].TryGetValue(cellUID, out nCell))
                {
                    var newCell = new MeshCell();

                    // Convert the grid index to a real location.
                    newCell.LocX = (idxX << cellSizeExp) + (cellSize * 0.5f);
                    newCell.LocY = (idxY << cellSizeExp) + (cellSize * 0.5f);

                    newCell.xID = idxX;
                    newCell.yID = idxY;

                    newCell.Size = cellSize;

                    newCell.Mass += body.Mass;
                    newCell.CmX += body.Mass * body.LocX;
                    newCell.CmY += body.Mass * body.LocY;

                    newCell.BodyCount = 1;
                    newCell.ID = cellIdx;
                    newCell.BodyStartIdx = cellIdx;

                    meshDict[0].Add(cellUID, newCell);
                    meshBods.Add(new List<int>() { b }); // Add body index to mesh-body collection.

                    bodies[b].MeshID = cellIdx; // Set the body mesh ID.

                    childIdx.Add(new List<int>());

                    cellIdx++;
                }
                else // Add body to existing cell.
                {
                    //var cell = meshDict[0][cellUID];

                    nCell.Mass += body.Mass;
                    nCell.CmX += body.Mass * body.LocX;
                    nCell.CmY += body.Mass * body.LocY;
                    nCell.BodyCount++;

                    meshDict[0][cellUID] = nCell;
                    meshBods[nCell.ID].Add(b); // Add body index to mesh-body collection.

                    bodies[b].MeshID = nCell.ID; // Set the body mesh ID.
                }
            }

            //// Get the completed mesh array from the dictionary for furthur processing.
            //var meshArr = meshDict[0].Values.ToArray();

            var meshList = meshDict[0].Values.ToList();

            // Calculate the final center of mass for each cell.
            for (int m = 0; m < meshList.Count; m++)
            {
                var cell = meshList[m];
                cell.CmX = cell.CmX / (float)cell.Mass;
                cell.CmY = cell.CmY / (float)cell.Mass;
                meshList[m] = cell;
            }

            // Build flattened mesh-body and mesh-neighbor indexes:
            //
            // Mesh cells contain a start index and a count which will be used
            // within the kernels to access the correct elements of these indexes.

            // Build the mesh-body index.
            _meshBodies = BuildMeshBodyIndex(ref meshList, meshBods, bodies.Length);

            int[] levelIdx = new int[_levels + 1];
            levelIdx[0] = 0;

            for (int i = 1; i <= _levels; i++)
            {
                BuildNextLevel(ref meshList, ref meshDict, ref childIdx, cellSizeExp, i, ref levelIdx);
            }

            _mesh = meshList.ToArray();

            _levelIdx = levelIdx;

            //  timer2.Restart();

            // Build the mesh-neighbor index.
            _meshNeighbors = BuildMeshNeighborIndex(ref _mesh, meshDict);

            //  Console.WriteLine($@"C/N: {timer2.Elapsed.Milliseconds}");

            _meshChilds = BuildMeshChildIndex(ref _mesh, childIdx);
        }

        private void BuildNextLevel(ref List<MeshCell> mesh, ref Dictionary<int, MeshCell>[] meshDict, ref List<List<int>> childIdx, int cellSizeExp, int level, ref int[] levelIdx)
        {
            cellSizeExp += level;

            meshDict[level] = new Dictionary<int, MeshCell>();

            int cellSize = (int)Math.Pow(2, cellSizeExp);
            // Dictionary to hold the current mesh cells for fast lookups.

            // Current cell index.
            int cellIdx = mesh.Count;
            levelIdx[level] = cellIdx;
            int start = cellIdx;

            for (int m = levelIdx[level - 1]; m < start; m++)
            {
                var cell = mesh[m];

                // Calculate the cell position from the current body position.
                // Right bit-shift to get the x/y grid indexes.

                int idxX = (int)cell.LocX >> cellSizeExp;
                int idxY = (int)cell.LocY >> cellSizeExp;

                // Interleave the x/y indexes to create a morton number; use this for cell UID/Hash.
                var cellUID = MortonNumber(idxX, idxY);

                MeshCell nCell;
                // Add body to new cell.
                if (!meshDict[level].TryGetValue(cellUID, out nCell))
                {
                    var newCell = new MeshCell();

                    // Convert the grid index to a real location.
                    newCell.LocX = (idxX << cellSizeExp) + (cellSize * 0.5f);
                    newCell.LocY = (idxY << cellSizeExp) + (cellSize * 0.5f);

                    newCell.xID = idxX;
                    newCell.yID = idxY;

                    newCell.Size = cellSize;

                    newCell.Mass += cell.Mass;
                    newCell.CmX += (float)cell.Mass * cell.CmX;
                    newCell.CmY += (float)cell.Mass * cell.CmY;

                    newCell.BodyCount = cell.BodyCount;
                    newCell.ID = cellIdx;
                    newCell.Level = level;

                    childIdx.Add(new List<int>() { cell.ID });

                    meshDict[level].Add(cellUID, newCell);

                    mesh.Add(newCell);

                    var mcell = mesh[m];
                    mcell.ParentID = cellIdx;
                    mesh[m] = mcell;

                    cellIdx++;
                }
                else
                {
                    int id = nCell.ID;
                    var pcell = mesh[id];

                    pcell.Mass += cell.Mass;
                    pcell.CmX += (float)cell.Mass * cell.CmX;
                    pcell.CmY += (float)cell.Mass * cell.CmY;
                    pcell.BodyCount += cell.BodyCount;

                    var mcell = mesh[m];
                    mcell.ParentID = pcell.ID;
                    mesh[m] = mcell;

                    meshDict[level][cellUID] = pcell;
                    mesh[id] = pcell;

                    childIdx[id].Add(cell.ID);
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
        /// Builds a flattened index of mesh bodies.
        /// </summary>
        /// <param name="mesh">Particle mesh array.</param>
        /// <param name="meshBods">List of containing a list of body indexes for each mesh cell.</param>
        /// <param name="bodyCount">Total number of bodies in the current field.</param>
        private int[] BuildMeshBodyIndex(ref List<MeshCell> mesh, List<List<int>> meshBods, int bodyCount)
        {
            // Collection to store the body indexes.
            // Initialized with current body count to reduce resizing overhead.
            var bodList = new List<int>(bodyCount);

            // Iterate the mesh cells.
            for (int m = 0; m < mesh.Count; m++)
            {
                var cell = mesh[m];
                // Set the body start index to the current body list count.
                cell.BodyStartIdx = bodList.Count;

                // Iterate the inner list to get the body indexes for the current cell.
                for (int b = 0; b < meshBods[m].Count; b++)
                {
                    // Add body index to the collection.
                    bodList.Add(meshBods[m][b]);
                }

                mesh[m] = cell;
            }

            // Return the flattened body index array.
            return bodList.ToArray();
        }

        private int[] BuildMeshChildIndex(ref MeshCell[] mesh, List<List<int>> childIdx)
        {
            // Collection to store the body indexes.
            // Initialized with current body count to reduce resizing overhead.
            var childList = new List<int>();

            // Iterate the mesh cells.
            for (int m = 0; m < mesh.Length; m++)
            {
                // Set the body start index to the current body list count.
                mesh[m].ChildIdxStart = childList.Count;
                mesh[m].ChildCount = childIdx[m].Count;
                // Iterate the inner list to get the body indexes for the current cell.
                for (int b = 0; b < childIdx[m].Count; b++)
                {
                    // Add body index to the collection.
                    childList.Add(childIdx[m][b]);
                }
            }

            // Return the flattened body index array.
            return childList.ToArray();
        }

        /// <summary>
        /// Builds a flattened index of mesh neighbors.
        /// </summary>
        /// <param name="mesh">Particle mesh array.</param>
        /// <param name="meshDict">Mesh cell and cell UID/Hash collection.</param>
        /// <param name="cellSize">Size of mesh cells.</param>
        private int[] BuildMeshNeighborIndex(ref MeshCell[] mesh, Dictionary<int, MeshCell>[] meshDict)
        {
            //  timer2.Restart();

            // Collection to store the the mesh neighbor indexes.
            // Initialized with mesh length * 9 (8 neighbors per cell plus itself).
            var neighborIdxList = new List<int>(mesh.Length * 9);
            var neighborLevels = new List<int>[_levels + 1];

            var counts = new int[mesh.Length];

            var meshCopy = new MeshCell[mesh.Length];
            Array.Copy(mesh, meshCopy, mesh.Length);

            var options = new ParallelOptions();
            options.MaxDegreeOfParallelism = _levels + 1;

            Parallel.For(0, _levels + 1, options, (level) =>
            {
                int len = 0;

                if (level < _levels)
                {
                    len = _levelIdx[level + 1];
                }
                else
                {
                    len = meshCopy.Length;
                }

                neighborLevels[level] = new List<int>();

                for (int i = _levelIdx[level]; i < len; i++)
                {
                    // Count of neighbors found.
                    int count = 0;

                    for (int x = -1; x <= 1; x++)
                    {
                        for (int y = -1; y <= 1; y++)
                        {
                            // Apply the current X/Y mulipliers to the mesh grid coords to get
                            // the coordinates of a neighboring cell.
                            int nX = meshCopy[i].xID + x;
                            int nY = meshCopy[i].yID + y;

                            // Convert the new coords to a cell UID/Hash and check if the cell exists.
                            var cellUID = MortonNumber(nX, nY);

                            MeshCell nCell;
                            if (meshDict[level].TryGetValue(cellUID, out nCell))
                            {
                                // Add the neighboring cell ID.
                                neighborLevels[level].Add(nCell.ID);
                                count++;
                            }
                        }
                    }

                    counts[i] = count;
                }
            });

            //  Console.WriteLine(timer2.ElapsedMilliseconds);

            int start = 0;

            for (int level = 0; level <= _levels; level++)
            {
                int len = 0;

                if (level < _levels)
                {
                    len = _levelIdx[level + 1];
                }
                else
                {
                    len = mesh.Length;
                }

                for (int i = _levelIdx[level]; i < len; i++)
                {
                    mesh[i].NeighborStartIdx = start;
                    mesh[i].NeighborCount = counts[i];

                    start += counts[i];
                }

                for (int n = 0; n < neighborLevels[level].Count; n++)
                {
                    neighborIdxList.Add(neighborLevels[level][n]);
                }
            }


            // Return the flattened mesh neighbor index.
            return neighborIdxList.ToArray();
        }

        /// <summary>
        /// Calculates the gravitational forces, and SPH density/pressure. Also does initial collision detection.
        /// </summary>
        [Cudafy]
        public static void CalcForce(GThread gpThread, Body[] inBodies, Body[] outBodies, MeshCell[] inMesh, int[] inMeshBods, int[] meshNeighbors, int[] meshChilds, float dt, int topLevel, int[] levelIdx)
        {
            float GAS_K = 0.3f;
            float FLOAT_EPSILON = 1.192092896e-07f;

            // SPH variables
            float ksize;
            float ksizeSq;
            float factor;
            float diff;
            float fac;

            float totMass;
            float force;
            float distX;
            float distY;
            float dist;
            float distSqrt;

            // Get index for the current body.
            int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            if (a > inBodies.Length - 1)
                return;

            // Copy current body and mesh cell from memory.
            Body outBody = inBodies[a];
            MeshCell bodyCell = inMesh[outBody.MeshID];
            MeshCell levelCell = bodyCell;
            MeshCell levelCellParent = inMesh[bodyCell.ParentID];

            // Reset forces.
            outBody.ForceTot = 0;
            outBody.ForceX = 0;
            outBody.ForceY = 0;
            outBody.HasCollision = 0;

            outBody.Density = 0;
            outBody.Pressure = 0;

            // Calculate initial (resting) body density.
            ksize = 1.0f;
            ksizeSq = 1.0f;
            factor = 1.566682f;

            fac = 1.566681f;
            outBody.Density = (outBody.Mass * fac);

            for (int level = 0; level < topLevel; level++)
            {
                int start = 0;
                int len = 0;

                start = levelCellParent.NeighborStartIdx;
                len = start + levelCellParent.NeighborCount;

                for (int nc = start; nc < len; nc++)
                {
                    int nId = meshNeighbors[nc];
                    MeshCell nCell = inMesh[nId];

                    int childStartIdx = nCell.ChildIdxStart;
                    int childLen = childStartIdx + nCell.ChildCount;

                    for (int c = childStartIdx; c < childLen; c++)
                    {
                        int cId = meshChilds[c];

                        // Make sure the current cell index is not a neighbor or this body's cell.
                        if (cId != outBody.MeshID)
                        {
                            // Calculate the force from the cells center of mass.
                            MeshCell cell = inMesh[cId];

                            if (IsNeighbor(levelCell, cell) == -1)
                            {
                                distX = cell.CmX - outBody.LocX;
                                distY = cell.CmY - outBody.LocY;
                                dist = (distX * distX) + (distY * distY);

                                distSqrt = (float)Math.Sqrt(dist);

                                totMass = (float)cell.Mass * outBody.Mass;
                                force = totMass / dist;

                                outBody.ForceTot += force;
                                outBody.ForceX += (force * distX / distSqrt);
                                outBody.ForceY += (force * distY / distSqrt);
                            }
                        }
                    }
                }

                levelCell = levelCellParent;
                levelCellParent = inMesh[levelCellParent.ParentID];
            }

            for (int top = levelIdx[topLevel]; top < inMesh.Length; top++)
            {
                MeshCell cell = inMesh[top];

                if (IsNeighbor(levelCell, cell) == -1)
                {
                    distX = cell.CmX - outBody.LocX;
                    distY = cell.CmY - outBody.LocY;
                    dist = (distX * distX) + (distY * distY);

                    distSqrt = (float)Math.Sqrt(dist);

                    totMass = (float)cell.Mass * outBody.Mass;
                    force = totMass / dist;

                    outBody.ForceTot += force;
                    outBody.ForceX += (force * distX / distSqrt);
                    outBody.ForceY += (force * distY / distSqrt);
                }
            }

            // Accumulate forces from all bodies within neighboring cells. [THIS INCLUDES THE BODY'S OWN CELL]
            // Read from the flattened mesh-neighbor index at the correct location.
            for (int n = bodyCell.NeighborStartIdx; n < bodyCell.NeighborStartIdx + bodyCell.NeighborCount; n++)
            {
                // Get the mesh cell index, then copy it from memory.
                int nId = meshNeighbors[n];
                MeshCell cell = inMesh[nId];

                // Iterate the bodies within the cell.
                // Read from the flattened mesh-body index at the correct location.
                int mbStart = cell.BodyStartIdx;
                int mbLen = cell.BodyCount + mbStart;
                for (int mb = mbStart; mb < mbLen; mb++)
                {
                    // Get the mesh body index, then copy it from memory.
                    int meshBodId = inMeshBods[mb];
                    Body inBody = inBodies[meshBodId];

                    // Save us from ourselves.
                    if (meshBodId != a)
                    {
                        distX = inBody.LocX - outBody.LocX;
                        distY = inBody.LocY - outBody.LocY;
                        dist = (distX * distX) + (distY * distY);

                        // If this body is within collision/SPH distance.
                        if (dist <= ksize)
                        {
                            // Set collision flag. Saves some cycles later in the collision kernel.
                            outBody.HasCollision = 1;

                            // Clamp SPH softening distance.
                            if (dist < FLOAT_EPSILON)
                            {
                                dist = FLOAT_EPSILON;
                            }

                            // Accumulate density.
                            diff = ksizeSq - dist;
                            fac = factor * diff * diff * diff;
                            outBody.Density += outBody.Mass * fac;
                        }

                        // Clamp gravity softening distance.
                        if (dist < 0.04f)
                        {
                            dist = 0.04f;
                        }

                        // Accumulate body-to-body force.
                        distSqrt = (float)Math.Sqrt(dist);

                        totMass = inBody.Mass * outBody.Mass;
                        force = totMass / dist;

                        outBody.ForceTot += force;
                        outBody.ForceX += (force * distX / distSqrt);
                        outBody.ForceY += (force * distY / distSqrt);
                    }
                }
            }

            gpThread.SyncThreads();

            // Calculate pressure from density.
            outBody.Pressure = GAS_K * (outBody.Density);

            if (outBody.ForceTot > outBody.Mass * 4 & outBody.BlackHole == 0)
            {
                outBody.InRoche = 1;
            }
            else if (outBody.ForceTot * 2 < outBody.Mass * 4)
            {
                outBody.InRoche = 0;
            }
            else if (outBody.BlackHole == 2 || outBody.IsExplosion == 1)
            {
                outBody.InRoche = 1;
            }

            if (outBody.BlackHole == 2)
                outBody.InRoche = 1;

            // Write back to memory.
            outBodies[a] = outBody;
        }

        /// <summary>
        /// Tests the specified cell index to see if it falls within the specified range of neighbor cell indexes.
        /// </summary>
        [Cudafy]
        public static int IsNeighbor(MeshCell testCell, MeshCell neighborCell)
        {
            int match = -1;

            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    if (neighborCell.xID == testCell.xID + x && neighborCell.yID == testCell.yID + y)
                        match = 1;
                }
            }

            return match;
        }
        
        /// <summary>
        /// Calculates elastic and SPH collision forces then integrates movement.
        /// </summary>
        [Cudafy]
        public static void CalcCollisions(GThread gpThread, Body[] inBodies, Body[] outBodies, MeshCell[] inMesh, int[] inMeshBods, int[] meshNeighbors, float dt, float viscosity, int drift)
        {
            float distX;
            float distY;
            float dist;
            float distSqrt;

            // Get index for the current body.
            int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            if (a > inBodies.Length - 1)
                return;

            // Copy current body from memory.
            Body outBody = inBodies[a];

            // Don't bother doing any work if there's no collision.
            if (outBody.HasCollision == 1)
            {
                // Copy this body's mesh cell from memory.
                MeshCell bodyCell = inMesh[outBody.MeshID];

                // Iterate neighbor cells.
                for (int i = bodyCell.NeighborStartIdx; i < bodyCell.NeighborStartIdx + bodyCell.NeighborCount; i++)
                {
                    // Get the neighbor cell from the index.
                    int nId = meshNeighbors[i];
                    MeshCell cell = inMesh[nId];

                    // Iterate the neighbor cell bodies.
                    int mbStart = cell.BodyStartIdx;
                    int mbLen = cell.BodyCount + mbStart;
                    for (int mb = mbStart; mb < mbLen; mb++)
                    {
                        // Get the cell body from the index.
                        int meshBodId = inMeshBods[mb];
                        Body inBody = inBodies[meshBodId];

                        // Double tests are bad.
                        if (inBody.UID != outBody.UID)
                        {
                            distX = outBody.LocX - inBody.LocX;
                            distY = outBody.LocY - inBody.LocY;
                            dist = (distX * distX) + (distY * distY);

                            // Calc the distance and check for collision.
                            float colDist = (outBody.Size * 0.5f) + (inBody.Size * 0.5f);
                            if (dist <= colDist * colDist)
                            {
                                // We know we have a collision, so go ahead and do the expensive square root now.
                                distSqrt = (float)Math.Sqrt(dist);

                                // If both bodies are in Roche, we do SPH physics.
                                // Otherwise, an elastic collision and merge is done.

                                // SPH collision.
                                if (outBody.InRoche == 1 & inBody.InRoche == 1)
                                {
                                    float FLOAT_EPSILON = 1.192092896e-07f;
                                    float FLOAT_EPSILONSQRT = 3.45267E-11f;
                                    float m_kernelSize = 1.0f;

                                    if (dist < FLOAT_EPSILON)
                                    {
                                        dist = FLOAT_EPSILON;
                                        distSqrt = FLOAT_EPSILONSQRT;
                                    }

                                    // Pressure force
                                    float scalar = outBody.Mass * (outBody.Pressure + inBody.Pressure) / (2.0f * outBody.Density);
                                    float gradFactor = -10442.157f * (m_kernelSize - distSqrt) * (m_kernelSize - distSqrt) / distSqrt;

                                    float gradX = (distX * gradFactor);
                                    float gradY = (distY * gradFactor);

                                    gradX = gradX * scalar;
                                    gradY = gradY * scalar;

                                    outBody.ForceX -= gradX;
                                    outBody.ForceY -= gradY;

                                    // Viscosity force
                                    float visc_Laplace = 14.323944f * (m_kernelSize - distSqrt);
                                    float visc_scalar = inBody.Mass * visc_Laplace * viscosity * 1.0f / inBody.Density;

                                    float viscVelo_diffX = inBody.SpeedX - outBody.SpeedX;
                                    float viscVelo_diffY = inBody.SpeedY - outBody.SpeedY;

                                    viscVelo_diffX *= visc_scalar;
                                    viscVelo_diffY *= visc_scalar;

                                    outBody.ForceX += viscVelo_diffX;
                                    outBody.ForceY += viscVelo_diffY;
                                }
                                // Elastic collision.
                                else if (outBody.InRoche == 1 & inBody.InRoche == 0) // Out of roche bodies always consume in roche bodies.
                                {
                                    outBody.Visible = 0; // Our body is merging with another body, somewhere in a far off thread.
                                }
                                else
                                {
                                    // Calculate elastic collision forces.
                                    float dotProd = distX * (inBody.SpeedX - outBody.SpeedX) + distY * (inBody.SpeedY - outBody.SpeedY);
                                    float colScale = dotProd / dist;
                                    float colForceX = distX * colScale;
                                    float colForceY = distY * colScale;
                                    float colMass = inBody.Mass / (inBody.Mass + outBody.Mass);

                                    // If we're the bigger one, eat the other guy.
                                    if (outBody.Mass > inBody.Mass)
                                    {
                                        outBody = CollideBodies(outBody, inBody, colMass, colForceX, colForceY);
                                    }
                                    else if (outBody.Mass < inBody.Mass) // We're smaller, so we must go away.
                                    {
                                        outBody.Visible = 0;
                                    }
                                    else if (outBody.Mass == inBody.Mass)  // If we are the same size, use a different metric.
                                    {
                                        // Our UID is more gooder, eat the other guy.
                                        if (outBody.UID > inBody.UID)
                                        {
                                            outBody = CollideBodies(outBody, inBody, colMass, colForceX, colForceY);
                                        }
                                        else // Our UID is inferior, we must go away.
                                        {
                                            outBody.Visible = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            gpThread.SyncThreads();

            // Leap frog integration.
            float dt2;
            float accelX;
            float accelY;

            // Drift
            if (drift == 1)
            {
                dt2 = dt * 0.5f;

                accelX = outBody.ForceX / outBody.Mass;
                accelY = outBody.ForceY / outBody.Mass;

                outBody.SpeedX += (accelX * dt2);
                outBody.SpeedY += (accelY * dt2);

                outBody.LocX += outBody.SpeedX * dt;
                outBody.LocY += outBody.SpeedY * dt;

                if (outBody.Lifetime > 0.0f)
                    outBody.Age += (dt * 4.0f);
            }
            else if (drift == 0) // Kick
            {
                dt2 = dt * 0.5f;

                accelX = outBody.ForceX / outBody.Mass;
                accelY = outBody.ForceY / outBody.Mass;

                outBody.SpeedX += accelX * dt2;
                outBody.SpeedY += accelY * dt2;
            }
            else if (drift == 3)  // Euler
            {
                outBody.SpeedX += dt * outBody.ForceX / outBody.Mass;
                outBody.SpeedY += dt * outBody.ForceY / outBody.Mass;
                outBody.LocX += dt * outBody.SpeedX;
                outBody.LocY += dt * outBody.SpeedY;

                if (outBody.Lifetime > 0.0f)
                    outBody.Age += (dt * 4.0f);
            }

            // Write back to memory.
            outBodies[a] = outBody;
        }

        [Cudafy]
        public static Body CollideBodies(Body master, Body slave, float colMass, float forceX, float forceY)
        {
            Body bodyA = master;
            Body bodyB = slave;

            bodyA.SpeedX += colMass * forceX;
            bodyA.SpeedY += colMass * forceY;

            if (bodyA.BlackHole != 1)
            {
                float a1 = (float)Math.PI * (float)(Math.Pow(bodyA.Size * 0.5f, 2));
                float a2 = (float)Math.PI * (float)(Math.Pow(bodyB.Size * 0.5f, 2));
                float a = a1 + a2;
                bodyA.Size = (float)Math.Sqrt((float)(a / Math.PI)) * 2;
            }

            bodyA.Mass += bodyB.Mass;

            return bodyA;
        }
    }
}