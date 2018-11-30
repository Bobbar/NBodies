using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Diagnostics;
using NBodies.Rendering;
using System.Linq;
using System.Collections.Generic;

namespace NBodies.Physics
{
    public class CUDAFloat : IPhysicsCalc
    {
        private int gpuIndex = 2;
        private static int threadsPerBlock = 256;
        private GPGPU gpu;
        private MeshCell[] _mesh;
        private MeshCell[] _rawMesh;
        private int[,] _meshBodies = new int[0, 0];

        public MeshCell[] CurrentMesh
        {
            get
            {
                return _mesh;
            }
        }

        public MeshCell[] RawMesh
        {
            get
            {
                return _rawMesh;
            }
        }

        public int[,] MeshBodies
        {
            get
            {
                return _meshBodies;
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
                threadsPerBlock = threadsperblock;
        }

        public void Init()
        {
            var cudaModule = CudafyModule.TryDeserialize();

            if (cudaModule == null || !cudaModule.TryVerifyChecksums())
            {
                CudafyTranslator.Language = eLanguage.OpenCL;
                cudaModule = CudafyTranslator.Cudafy(new Type[] { typeof(Body), typeof(MeshCell), typeof(CUDAFloat) });
                cudaModule.Serialize();
            }

            // Add missing 'struct' strings to generated code.
            cudaModule.SourceCode = FixCode(cudaModule.SourceCode);
            cudaModule.Serialize();

            gpu = CudafyHost.GetDevice(eGPUType.OpenCL, gpuIndex);
            gpu.LoadModule(cudaModule);



            var props = gpu.GetDeviceProperties();
            Console.WriteLine(props.ToString());
        }



        /// <summary>
        /// Fixes missing 'struct' strings for each Body function and variable declaration.
        /// </summary>
        private string FixCode(string code)
        {
            string newcode = string.Copy(code);

            bool missingDec = true;

            int lastIdx = 0;

            while (missingDec)
            {
                int idx = newcode.IndexOf(nameof(Body), lastIdx);

                if (idx == -1)
                {
                    missingDec = false;
                    continue;
                }

                lastIdx = idx + 1;

                string sub = newcode.Substring(idx - 7, 7);

                if (!sub.Contains("struct"))
                {
                    newcode = newcode.Insert(idx, "struct ");
                }
            }

            missingDec = true;
            lastIdx = 0;

            while (missingDec)
            {
                int idx = newcode.IndexOf(nameof(MeshCell), lastIdx);

                if (idx == -1)
                {
                    missingDec = false;
                    continue;
                }

                lastIdx = idx + 1;

                string sub = newcode.Substring(idx - 7, 7);

                if (!sub.Contains("struct"))
                {
                    newcode = newcode.Insert(idx, "struct ");
                }
            }


            return newcode;
        }

        public void CalcMovement(ref Body[] bodies, float timestep)
        {
            float viscosity = 10.0f;//20.0f;//40.0f;//5.0f;//7.5f;

            int blocks = 0;

            gpu.StartTimer();

            //_mesh = GetNewMesh(bodies);

            _mesh = BuildMeshTest(ref bodies, 30f);

            //var meshCopy = new MeshCell[_mesh.Length];
            //Array.Copy(_mesh, meshCopy, _mesh.Length);

            //_rawMesh = meshCopy;

            //blocks = BlockCount(_mesh.Length);

            //gpu.FreeAll();

            //var gpuInBodiesMesh = gpu.Allocate(bodies);
            //var gpuInMesh = gpu.Allocate(_mesh);
            //var gpuOutMesh = gpu.Allocate(_mesh);
            //int[] meshBods = Enumerable.Repeat(-1, bodies.Length).ToArray();
            //var gpuOutMeshBods = gpu.Allocate(meshBods);

            //gpu.CopyToDevice(bodies, gpuInBodiesMesh);
            //gpu.CopyToDevice(_mesh, gpuInMesh);
            //gpu.CopyToDevice(meshBods, gpuOutMeshBods);

            //gpu.Launch(blocks, threadsPerBlock).PopulateMesh(gpuInBodiesMesh, gpuInMesh, gpuOutMesh, gpuOutMeshBods);

            //gpu.CopyFromDevice(gpuOutMesh, _mesh);
            //gpu.CopyFromDevice(gpuOutMeshBods, meshBods);

            //gpu.FreeAll();

            Console.WriteLine("Populate: " + gpu.StopTimer());


            // gpu.StartTimer();

            //// ShrinkMesh(_mesh, meshBods, ref bodies);

            // Console.WriteLine("Prep: " + gpu.StopTimer());


            if (_mesh.Length != _meshBodies.GetLength(0))
            {
                Debugger.Break();
            }

            blocks = BlockCount(bodies.Length);

            var gpuMesh = gpu.Allocate(_mesh);
            var gpuMeshBodies = gpu.Allocate(_meshBodies);
            var gpuInBodies = gpu.Allocate(bodies);
            var gpuOutBodies = gpu.Allocate(bodies);

            gpu.CopyToDevice(bodies, gpuInBodies);
            gpu.CopyToDevice(_mesh, gpuMesh);
            gpu.CopyToDevice(_meshBodies, gpuMeshBodies);

            gpu.StartTimer();

            if (MainLoop.LeapFrog)
            {
                for (int drift = 1; drift > -1; drift--)
                {
                    gpu.Launch(blocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, gpuMesh, gpuMeshBodies, timestep);
                    gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, gpuMesh, gpuMeshBodies, timestep, viscosity, drift);
                }
            }
            else
            {
                gpu.Launch(blocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, gpuMesh, gpuMeshBodies, timestep);
                gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, gpuMesh, gpuMeshBodies, timestep, viscosity, 3);
            }

            Console.WriteLine("Kern: " + gpu.StopTimer());

            gpu.CopyFromDevice(gpuInBodies, bodies);

            gpu.FreeAll();
        }

        public static int BlockCount(int len, int threads = 0)
        {
            if (threads == 0)
                threads = threadsPerBlock;

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

        private MeshCell[] GetNewMesh(Body[] bodies)
        {
            float cellSize = 30f;

            var maxX = bodies.Max(b => b.LocX) + cellSize;
            var minX = bodies.Min(b => b.LocX);

            var maxY = bodies.Max(b => b.LocY) + cellSize;
            var minY = bodies.Min(b => b.LocY);


            Console.WriteLine($@"MinX: { minX } MaxX: { maxX } ");
            Console.WriteLine($@"MinY: { minY } MaxY: { maxY } ");

            int cellsToOrig = 0;

            cellsToOrig = (int)(maxX / cellSize);
            if ((cellsToOrig * cellSize) < maxX)
            {
                maxX = cellsToOrig * cellSize + cellSize;
            }
            else
            {
                maxX = cellsToOrig * cellSize;
            }

            cellsToOrig = (int)(minX / cellSize);
            if ((cellsToOrig * cellSize) > minX)
            {
                minX = cellsToOrig * cellSize - cellSize;
            }
            else
            {
                minX = cellsToOrig * cellSize;
            }

            cellsToOrig = (int)(maxY / cellSize);
            if ((cellsToOrig * cellSize) < maxY)
            {
                maxY = cellsToOrig * cellSize + cellSize;
            }
            else
            {
                maxY = cellsToOrig * cellSize;
            }

            cellsToOrig = (int)(minY / cellSize);
            if ((cellsToOrig * cellSize) > minY)
            {
                minY = cellsToOrig * cellSize - cellSize;
            }
            else
            {
                minY = cellsToOrig * cellSize;
            }

            Console.WriteLine($@"MinX: { minX } MaxX: { maxX } ");
            Console.WriteLine($@"MinY: { minY } MaxY: { maxY } ");

            float cellRadius = 0;
            //int nodes = 20; //100;
            int cellsPerRow = 0; //100;

            float meshSize = 0;

            var wX = maxX - minX;
            var wY = maxY - minY;

            if (wX > wY)
            {
                meshSize = wX;
            }
            else
            {
                meshSize = wY;
            }

            cellsPerRow = (int)(meshSize / cellSize);
            cellsPerRow++;

            cellRadius = cellSize / 2f;

            float curX = minX;
            float curY = minY;

            MeshCell[] mesh = new MeshCell[cellsPerRow * cellsPerRow];

            int rightSteps = 0;
            int downSteps = -1;

            for (int i = 0; i < cellsPerRow * cellsPerRow; i++)
            {
                mesh[i].LocX = curX;
                mesh[i].LocY = curY;
                mesh[i].Mass = 0f;
                mesh[i].BodCount = 0;
                mesh[i].Size = cellSize;

                mesh[i].Top = mesh[i].LocY + cellRadius;
                mesh[i].Bottom = mesh[i].LocY - cellRadius;
                mesh[i].Left = mesh[i].LocX - cellRadius;
                mesh[i].Right = mesh[i].LocX + cellRadius;

                curX += cellSize;
                rightSteps++;

                if (rightSteps == cellsPerRow)
                {
                    curX = minX;
                    rightSteps = 0;

                    curY += cellSize;
                    downSteps++;
                }

            }

            return mesh;
        }

        private MeshCell[] BuildMeshTest(ref Body[] bodies, float cellSize)
        {
            List<MeshCell> mesh = new List<MeshCell>();
            int[] meshBodies = new int[bodies.Length];

            for (int b = 0; b < bodies.Length; b++)
            {
                var body = bodies[b];

                int cellIdx = FindCellIdx(mesh, body);

                if (cellIdx != -1)
                {
                    var cell = mesh[cellIdx];

                    cell.Mass += body.Mass;
                    cell.CmX += body.Mass * body.LocX;
                    cell.CmY += body.Mass * body.LocY;
                    cell.BodCount++;

                    mesh[cellIdx] = cell;
                    meshBodies[b] = cellIdx;
                }
                else
                {
                    var newCell = new MeshCell();

                    float cellX;
                    float cellY;
                    int cellsToOrig;
                    float edge;
                    float cellRad = cellSize / 2f;

                    if (body.LocX < 0)
                    {
                        cellsToOrig = (int)(body.LocX / cellSize);
                        edge = cellsToOrig * cellSize;
                        if ((edge - cellRad) > body.LocX)
                        {
                            cellX = edge - cellSize;
                        }
                        else
                        {
                            cellX = edge;
                        }

                    }
                    else
                    {
                        cellsToOrig = (int)(body.LocX / cellSize);
                        edge = cellsToOrig * cellSize;
                        if ((edge + cellRad) < body.LocX)
                        {
                            cellX = edge + cellSize;
                        }
                        else
                        {
                            cellX = edge;
                        }
                    }


                    if (body.LocY < 0)
                    {
                        cellsToOrig = (int)(body.LocY / cellSize);
                        edge = cellsToOrig * cellSize;
                        if ((edge - cellRad) > body.LocY)
                        {
                            cellY = edge - cellSize;
                        }
                        else
                        {
                            cellY = edge;
                        }
                    }
                    else
                    {
                        cellsToOrig = (int)(body.LocY / cellSize);
                        edge = cellsToOrig * cellSize;
                        if ((edge + cellRad) < body.LocY)
                        {
                            cellY = edge + cellSize;
                        }
                        else
                        {
                            cellY = edge;
                        }
                    }

                    newCell.LocX = cellX;
                    newCell.LocY = cellY;
                    newCell.Size = cellSize;

                    newCell.Left = newCell.LocX - cellRad;
                    newCell.Right = newCell.LocX + cellRad;
                    newCell.Top = newCell.LocY - cellRad;
                    newCell.Bottom = newCell.LocY + cellRad;

                    newCell.Mass += body.Mass;
                    newCell.CmX += body.Mass * body.LocX;
                    newCell.CmY += body.Mass * body.LocY;
                    newCell.BodCount = 1;

                    mesh.Add(newCell);

                    meshBodies[b] = mesh.Count - 1;
                }

            }

            var meshArr = mesh.ToArray();

            for (int m = 0; m < meshArr.Length; m++)
            {
                meshArr[m].CmX = meshArr[m].CmX / (float)meshArr[m].Mass;
                meshArr[m].CmY = meshArr[m].CmY / (float)meshArr[m].Mass;
            }

            ParseBodList(meshArr, meshBodies, ref bodies);

            return meshArr;

        }

        private int FindCellIdx(List<MeshCell> mesh, Body body)
        {
            for (int c = 0; c < mesh.Count; c++)
            {
                if (body.LocX < mesh[c].Right && body.LocX > mesh[c].Left && body.LocY > mesh[c].Top && body.LocY < mesh[c].Bottom)
                {
                    return c;
                }
            }

            return -1;
        }

        //private MeshCell FindCell(List<MeshCell> mesh, Body body)
        //{
        //    for (int c = 0; c < mesh.Count; c++)
        //    {
        //        if (body.LocX < mesh[c].Right && body.LocX > mesh[c].Left && body.LocY > mesh[c].Top && body.LocY < mesh[c].Bottom)
        //        {
        //            return mesh[c];
        //        }
        //    }

        //    var noMesh = new MeshCell();
        //    noMesh.BodCount = -1;
        //    return noMesh;
        //}

        private void ParseBodList(MeshCell[] mesh, int[] meshBods, ref Body[] bodies)
        {
            var meshBodDict = new Dictionary<int, int[]>();
            var meshBodList = new List<int[]>();
            int maxCount = mesh.Max(m => m.BodCount);

            // Since the body index array must be a fixed length, 
            // we need to track the current element index for each mesh.
            int[] curIdxCount = new int[mesh.Length]; // Number of body indexes added to each mesh.

            // Populate a dictionary with the Mesh indexes and an array of indexes for their contained Bodies.
            // Key = MeshID
            // Value = int[] of body indexes
            for (int b = 0; b < meshBods.Length; b++)
            {
                if (meshBods[b] != -1)
                {
                    if (!meshBodDict.ContainsKey(meshBods[b]))
                    {
                        meshBodDict.Add(meshBods[b], Enumerable.Repeat(-1, maxCount).ToArray());
                    }
                    // Add the body index to the dictionary at the current index.     
                    var idx = curIdxCount[meshBods[b]];
                    meshBodDict[meshBods[b]][idx] = b;
                    curIdxCount[meshBods[b]]++; // Increment the current index.
                }
            }

            // Itereate the meshes and collect only the ones which contain bodies.
            // Also builds the mesh/body index list.
            for (int m = 0; m < mesh.Length; m++)
            {
                if (mesh[m].BodCount > 0)
                {
                    meshBodList.Add(meshBodDict[m]); // Build a list of body indexes using the dictionary from above.
                }
            }

            // Iterate the mesh/body index list and update each body with their mesh ID.
            for (int i = 0; i < meshBodList.Count; i++)
            {
                foreach (int bodIdx in meshBodList[i])
                {
                    if (bodIdx != -1)
                    {
                        bodies[bodIdx].MeshID = i;
                    }
                }
            }

            _meshBodies = CreateRectangularArray(meshBodList);
        }


        private void ShrinkMesh(MeshCell[] meshes, int[] meshBods, ref Body[] bodies)
        {
            var meshList = new List<MeshCell>();
            var meshBodDict = new Dictionary<int, int[]>();
            var meshBodList = new List<int[]>();
            int maxCount = meshes.Max(m => m.BodCount);

            // Since the body index array must be a fixed length, 
            // we need to track the current element index for each mesh.
            int[] curIdxCount = new int[meshes.Length]; // Number of body indexes added to each mesh.

            // Populate a dictionary with the Mesh indexes and an array of indexes for their contained Bodies.
            // Key = MeshID
            // Value = int[] of body indexes
            for (int b = 0; b < meshBods.Length; b++)
            {
                if (meshBods[b] != -1)
                {
                    if (!meshBodDict.ContainsKey(meshBods[b]))
                    {
                        meshBodDict.Add(meshBods[b], Enumerable.Repeat(-1, maxCount).ToArray());
                    }
                    // Add the body index to the dictionary at the current index.     
                    var idx = curIdxCount[meshBods[b]];
                    meshBodDict[meshBods[b]][idx] = b;
                    curIdxCount[meshBods[b]]++; // Increment the current index.
                }
            }

            // Itereate the meshes and collect only the ones which contain bodies.
            // Also builds the mesh/body index list.
            for (int m = 0; m < meshes.Length; m++)
            {
                if (meshes[m].BodCount > 0)
                {
                    meshList.Add(meshes[m]);
                    meshBodList.Add(meshBodDict[m]); // Build a list of body indexes using the dictionary from above.
                }
            }

            // Iterate the mesh/body index list and update each body with their mesh ID.
            for (int i = 0; i < meshBodList.Count; i++)
            {
                foreach (int bodIdx in meshBodList[i])
                {
                    if (bodIdx != -1)
                    {
                        bodies[bodIdx].MeshID = i;
                    }
                }
            }

            // Turn the mesh list into and array, and convert the mesh/body index list into a GPU compatible 2d array.
            _mesh = meshList.ToArray();
            _meshBodies = CreateRectangularArray(meshBodList);//, maxCount);
        }

        // Converts a List<int[]> into a 2d array.
        // Credit: https://stackoverflow.com/a/9775057
        private T[,] CreateRectangularArray<T>(IList<T[]> arrays)
        {
            int minorLength = arrays[0].Length;
            T[,] ret = new T[arrays.Count, minorLength];
            for (int i = 0; i < arrays.Count; i++)
            {
                var array = arrays[i];
                if (array.Length != minorLength)
                {
                    throw new ArgumentException
                        ("All arrays must be the same length");
                }
                for (int j = 0; j < minorLength; j++)
                {
                    ret[i, j] = array[j];
                }
            }
            return ret;
        }

        [Cudafy]
        public static void PopulateMesh(GThread gpThread, Body[] inBodies, MeshCell[] inMeshes, MeshCell[] outMeshes, int[] outMeshBods)
        {
            int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            if (a > inMeshes.Length)
                return;

            MeshCell outMesh = inMeshes[a];

            for (int b = 0; b < inBodies.Length; b++)
            {
                Body body = inBodies[b];

                if (body.LocX < outMesh.Right && body.LocX > outMesh.Left && body.LocY < outMesh.Top && body.LocY > outMesh.Bottom)
                {
                    outMesh.Mass += body.Mass;
                    outMesh.CmX += body.Mass * body.LocX;
                    outMesh.CmY += body.Mass * body.LocY;
                    outMesh.BodCount++;

                    if (outMeshBods[b] == -1)
                    {
                        outMeshBods[b] = a;
                    }
                }

            }

            gpThread.SyncThreads();

            if (outMesh.BodCount > 0)
            {
                outMesh.CmX = outMesh.CmX / (float)outMesh.Mass;
                outMesh.CmY = outMesh.CmY / (float)outMesh.Mass;
            }


            outMeshes[a] = outMesh;
        }

        [Cudafy]
        public static void CalcForce(GThread gpThread, Body[] inBodies, Body[] outBodies, MeshCell[] inMesh, int[,] inMeshBods, float dt)
        {
            float GAS_K = 0.3f;//0.8f;// 0.1f
            float FLOAT_EPSILON = 1.192092896e-07f;

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

            int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            if (a > inBodies.Length)
                return;

            Body outBody = inBodies[a];

            outBody.ForceTot = 0;
            outBody.ForceX = 0;
            outBody.ForceY = 0;
            outBody.HasCollision = 0;

            outBody.Density = 0;
            outBody.Pressure = 0;

            ksize = 1.0f;
            ksizeSq = 1.0f;
            factor = 1.566682f;

            // Calculate initial body density.
            fac = 1.566681f;
            outBody.Density = (outBody.Mass * fac);

            int len = inMesh.Length;
            for (int b = 0; b < len; b++)
            {
                if (b != outBody.MeshID)
                {
                    MeshCell mesh = inMesh[b];

                    if (mesh.BodCount > 0)
                    {
                        distX = mesh.LocX - outBody.LocX;
                        distY = mesh.LocY - outBody.LocY;
                        dist = (distX * distX) + (distY * distY);

                        float maxDist = 7.8f;//20.0f;

                        if (dist > maxDist * maxDist)
                        {
                            distX = mesh.CmX - outBody.LocX;
                            distY = mesh.CmY - outBody.LocY;
                            dist = (distX * distX) + (distY * distY);

                            distSqrt = (float)Math.Sqrt(dist);

                            totMass = (float)mesh.Mass * outBody.Mass;
                            force = totMass / dist;

                            outBody.ForceTot += force;
                            outBody.ForceX += (force * distX / distSqrt);
                            outBody.ForceY += (force * distY / distSqrt);

                        }
                        else
                        {
                            for (int mb = 0; mb < inMeshBods.GetLength(1); mb++)
                            {
                                int meshBodId = inMeshBods[b, mb];

                                if (meshBodId != -1)
                                {
                                    Body inBody = inBodies[meshBodId];

                                    if (inBody.UID != outBody.UID)
                                    {
                                        distX = inBody.LocX - outBody.LocX;
                                        distY = inBody.LocY - outBody.LocY;
                                        dist = (distX * distX) + (distY * distY);

                                        if (dist < 0.04f)
                                        {
                                            dist = 0.04f;
                                        }

                                        distSqrt = (float)Math.Sqrt(dist);

                                        totMass = inBody.Mass * outBody.Mass;
                                        force = totMass / dist;

                                        outBody.ForceTot += force;
                                        outBody.ForceX += (force * distX / distSqrt);
                                        outBody.ForceY += (force * distY / distSqrt);

                                        float colDist = (outBody.Size) + (inBody.Size);
                                        if (dist <= colDist * colDist)
                                        {
                                            outBody.HasCollision = 1;
                                        }

                                        if (dist <= ksize)
                                        {
                                            if (dist < FLOAT_EPSILON)
                                            {
                                                dist = FLOAT_EPSILON;
                                            }

                                            //  It's a neighbor; accumulate density.
                                            diff = ksizeSq - (float)dist;
                                            fac = factor * diff * diff * diff;
                                            outBody.Density += outBody.Mass * fac;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

            }

            gpThread.SyncThreads();

            // Calc SPH pressures and gravity for bodies within this mesh node.
            for (int mb = 0; mb < inMeshBods.GetLength(1); mb++)
            {
                int meshBodId = inMeshBods[outBody.MeshID, mb];

                if (meshBodId != -1)
                {
                    Body inBody = inBodies[meshBodId];

                    if (inBody.UID != outBody.UID)
                    {
                        distX = inBody.LocX - outBody.LocX;
                        distY = inBody.LocY - outBody.LocY;
                        dist = (distX * distX) + (distY * distY);

                        if (dist < 0.04f)
                        {
                            dist = 0.04f;
                        }

                        distSqrt = (float)Math.Sqrt(dist);

                        totMass = inBody.Mass * outBody.Mass;
                        force = totMass / dist;

                        outBody.ForceTot += force;
                        outBody.ForceX += (force * distX / distSqrt);
                        outBody.ForceY += (force * distY / distSqrt);



                        float colDist = (outBody.Size) + (inBody.Size);
                        if (dist <= colDist * colDist)
                        {
                            outBody.HasCollision = 1;
                        }

                        // SPH Density Kernel
                        if (outBody.InRoche == 1 && inBody.InRoche == 1)
                        {
                            // is this distance close enough for kernal / neighbor calcs ?
                            if (dist <= ksize)
                            {
                                if (dist < FLOAT_EPSILON)
                                {
                                    dist = FLOAT_EPSILON;
                                }

                                //  It's a neighbor; accumulate density.
                                diff = ksizeSq - (float)dist;
                                fac = factor * diff * diff * diff;
                                outBody.Density += outBody.Mass * fac;
                            }
                        }
                    }
                }
            }

            gpThread.SyncThreads();

            outBody.Pressure = GAS_K * (outBody.Density);

            outBodies[a] = outBody;
        }

        [Cudafy]
        public static void CalcCollisions(GThread gpThread, Body[] inBodies, Body[] outBodies, MeshCell[] inMesh, int[,] inMeshBods, float dt, float viscosity, int drift)
        {
            float distX;
            float distY;
            float dist;
            float distSqrt;

            int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            if (a > inBodies.Length)
                return;

            Body outBody = inBodies[a];

            if (outBody.HasCollision == 1)
            {
                float FLOAT_EPSILON = 1.192092896e-07f;
                float FLOAT_EPSILONSQRT = 3.45267E-11f;
                float m_kernelSize = 1.0f;

                int len = inMesh.Length;
                for (int b = 0; b < len; b++)
                {
                    if (b != outBody.MeshID)
                    {
                        MeshCell mesh = inMesh[b];

                        if (mesh.BodCount > 0)
                        {
                            distX = mesh.LocX - outBody.LocX;
                            distY = mesh.LocY - outBody.LocY;
                            dist = (distX * distX) + (distY * distY);

                            float maxDist = 7.8f;

                            if (dist < maxDist * maxDist)
                            {
                                for (int mb = 0; mb < inMeshBods.GetLength(1); mb++)
                                {
                                    int meshBodId = inMeshBods[b, mb];

                                    if (meshBodId != -1)
                                    {
                                        Body inBody = inBodies[meshBodId];

                                        if (inBody.UID != outBody.UID)
                                        {
                                            distX = inBody.LocX - outBody.LocX;
                                            distY = inBody.LocY - outBody.LocY;
                                            dist = (distX * distX) + (distY * distY);

                                            float colDist = (outBody.Size * 0.5f) + (inBody.Size * 0.5f);
                                            if (dist <= colDist * colDist)
                                            {
                                                distSqrt = (float)Math.Sqrt(dist);

                                                if (outBody.InRoche == 1 & inBody.InRoche == 1)
                                                {
                                                    if (dist < FLOAT_EPSILON)
                                                    {
                                                        dist = FLOAT_EPSILON;
                                                        distSqrt = FLOAT_EPSILONSQRT;
                                                    }

                                                    float scalar = outBody.Mass * (outBody.Pressure + inBody.Pressure) / (2.0f * outBody.Density);
                                                    float gradFactor = -10442.157f * (m_kernelSize - distSqrt) * (m_kernelSize - distSqrt) / distSqrt;

                                                    float gradX = (distX * gradFactor);
                                                    float gradY = (distY * gradFactor);

                                                    gradX = gradX * scalar;
                                                    gradY = gradY * scalar;

                                                    outBody.ForceX += gradX;
                                                    outBody.ForceY += gradY;

                                                    // Viscosity
                                                    float visc_Laplace = 14.323944f * (m_kernelSize - distSqrt);
                                                    float visc_scalar = outBody.Mass * visc_Laplace * viscosity * 1.0f / outBody.Density;

                                                    float viscVelo_diffX = outBody.SpeedX - inBody.SpeedX;
                                                    float viscVelo_diffY = outBody.SpeedY - inBody.SpeedY;

                                                    viscVelo_diffX *= visc_scalar;
                                                    viscVelo_diffY *= visc_scalar;

                                                    outBody.ForceX -= viscVelo_diffX;
                                                    outBody.ForceY -= viscVelo_diffY;

                                                }
                                                else if (outBody.InRoche == 1 & inBody.InRoche == 0)
                                                {
                                                    outBody.Visible = 0;
                                                }
                                                else
                                                {
                                                    float dotProd = distX * (inBody.SpeedX - outBody.SpeedX) + distY * (inBody.SpeedY - outBody.SpeedY);
                                                    float colScale = dotProd / dist;
                                                    float colForceX = distX * colScale;
                                                    float colForceY = distY * colScale;
                                                    float colMass = inBody.Mass / (inBody.Mass + outBody.Mass);

                                                    if (outBody.Mass > inBody.Mass)
                                                    {
                                                        outBody = CollideBodies(outBody, inBody, colMass, colForceX, colForceY);
                                                    }
                                                    else if (outBody.Mass == inBody.Mass)
                                                    {
                                                        if (outBody.UID > inBody.UID)
                                                        {
                                                            outBody = CollideBodies(outBody, inBody, colMass, colForceX, colForceY);
                                                        }
                                                        else
                                                        {
                                                            outBody.Visible = 0;
                                                        }
                                                    }
                                                    else
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
                    }
                }

                gpThread.SyncThreads();

                for (int mb = 0; mb < inMeshBods.GetLength(1); mb++)
                {
                    int meshBodId = inMeshBods[outBody.MeshID, mb];

                    if (meshBodId != -1)
                    {
                        Body inBody = inBodies[meshBodId];

                        if (inBody.UID != outBody.UID)
                        {
                            distX = inBody.LocX - outBody.LocX;
                            distY = inBody.LocY - outBody.LocY;
                            dist = (distX * distX) + (distY * distY);

                            float colDist = (outBody.Size * 0.5f) + (inBody.Size * 0.5f);
                            if (dist <= colDist * colDist)
                            {
                                distSqrt = (float)Math.Sqrt(dist);

                                if (outBody.InRoche == 1 & inBody.InRoche == 1)
                                {
                                    if (dist < FLOAT_EPSILON)
                                    {
                                        dist = FLOAT_EPSILON;
                                        distSqrt = FLOAT_EPSILONSQRT;
                                    }

                                    float scalar = outBody.Mass * (outBody.Pressure + inBody.Pressure) / (2.0f * outBody.Density);
                                    float gradFactor = -10442.157f * (m_kernelSize - distSqrt) * (m_kernelSize - distSqrt) / distSqrt;

                                    float gradX = (distX * gradFactor);
                                    float gradY = (distY * gradFactor);

                                    gradX = gradX * scalar;
                                    gradY = gradY * scalar;

                                    outBody.ForceX += gradX;
                                    outBody.ForceY += gradY;

                                    // Viscosity
                                    float visc_Laplace = 14.323944f * (m_kernelSize - distSqrt);
                                    float visc_scalar = outBody.Mass * visc_Laplace * viscosity * 1.0f / outBody.Density;

                                    float viscVelo_diffX = outBody.SpeedX - inBody.SpeedX;
                                    float viscVelo_diffY = outBody.SpeedY - inBody.SpeedY;

                                    viscVelo_diffX *= visc_scalar;
                                    viscVelo_diffY *= visc_scalar;

                                    outBody.ForceX -= viscVelo_diffX;
                                    outBody.ForceY -= viscVelo_diffY;

                                }
                                else if (outBody.InRoche == 1 & inBody.InRoche == 0)
                                {
                                    outBody.Visible = 0;
                                }
                                else
                                {
                                    float dotProd = distX * (inBody.SpeedX - outBody.SpeedX) + distY * (inBody.SpeedY - outBody.SpeedY);
                                    float colScale = dotProd / dist;
                                    float colForceX = distX * colScale;
                                    float colForceY = distY * colScale;
                                    float colMass = inBody.Mass / (inBody.Mass + outBody.Mass);

                                    if (outBody.Mass > inBody.Mass)
                                    {
                                        outBody = CollideBodies(outBody, inBody, colMass, colForceX, colForceY);
                                    }
                                    else if (outBody.Mass == inBody.Mass)
                                    {
                                        if (outBody.UID > inBody.UID)
                                        {
                                            outBody = CollideBodies(outBody, inBody, colMass, colForceX, colForceY);
                                        }
                                        else
                                        {
                                            outBody.Visible = 0;
                                        }
                                    }
                                    else
                                    {
                                        outBody.Visible = 0;
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

            //  gpThread.SyncThreads();

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
                bodyA.Size = (float)Math.Sqrt(a / Math.PI) * 2;
            }

            bodyA.Mass += bodyB.Mass;

            return bodyA;
        }

    }
}