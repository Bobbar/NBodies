using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using NBodies.Rendering;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace NBodies.Physics
{
    public class CUDAFloat : IPhysicsCalc
    {
        private int gpuIndex = 2;
        private static int threadsPerBlock = 256;
        private GPGPU gpu;
        private MeshCell[] _mesh;
        private int[,] _meshBodies = new int[0, 0];
        private int[,] _meshNeighbors = new int[0, 0];

        public MeshCell[] CurrentMesh
        {
            get
            {
                return _mesh;
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

            //Add missing 'struct' strings to generated code.
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

        private static Stopwatch timer = new Stopwatch();

        public void CalcMovement(ref Body[] bodies, float timestep, float cellSize)
        {
            float viscosity = 10.0f;//20.0f;//40.0f;//5.0f;//7.5f;
            //float cellSize = 5f;//30f;

            int blocks = 0;

            timer.Restart();

            _mesh = BuildMesh(ref bodies, cellSize);

            // Console.WriteLine($@"{timer.ElapsedMilliseconds}");
            Console.WriteLine($@"Build ({_mesh.Length}): {timer.ElapsedMilliseconds}");

            blocks = BlockCount(bodies.Length);

            var gpuMesh = gpu.Allocate(_mesh);
            var gpuMeshBodies = gpu.Allocate(_meshBodies);
            var gpuMeshNeighbors = gpu.Allocate(_meshNeighbors);
            var gpuInBodies = gpu.Allocate(bodies);
            var gpuOutBodies = gpu.Allocate(bodies);

            gpu.CopyToDevice(bodies, gpuInBodies);
            gpu.CopyToDevice(_mesh, gpuMesh);
            gpu.CopyToDevice(_meshBodies, gpuMeshBodies);
            gpu.CopyToDevice(_meshNeighbors, gpuMeshNeighbors);


            gpu.StartTimer();

            if (MainLoop.LeapFrog)
            {
                for (int drift = 1; drift > -1; drift--)
                {
                    gpu.Launch(blocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, gpuMesh, gpuMeshBodies, gpuMeshNeighbors, timestep);
                    gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, gpuMesh, gpuMeshBodies, gpuMeshNeighbors, timestep, viscosity, drift);
                }
            }
            else
            {
                gpu.Launch(blocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, gpuMesh, gpuMeshBodies, gpuMeshNeighbors, timestep);
                gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, gpuMesh, gpuMeshBodies, gpuMeshNeighbors, timestep, viscosity, 3);
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

        static int[] B = new int[] { 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF };
        static int[] S = new int[] { 1, 2, 4, 8 };

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
        /// Builds the particle mesh and the mesh-body index for the current field.
        /// </summary>
        /// <param name="bodies">Array of bodies.</param>
        /// <param name="cellSize">The width/height of each cell in the mesh.</param>
        private MeshCell[] BuildMesh(ref Body[] bodies, float cellSize)
        {
            // Dictionary to hold the current mesh cells for fast lookups.
            var meshDict = new Dictionary<int, MeshCell>();

            // Current cell index.
            int cellIdx = 0;

            for (int b = 0; b < bodies.Length; b++)
            {
                // Calculate the cell position from the current body position.
                // Cell Pos = Round(Body Pos / cellSize) * cellSize
                // This formula aligns the mesh cells with the 0,0 origin.

                var body = bodies[b];

                int cellX;
                int cellY;

                float divX = body.LocX / cellSize;
                float divY = body.LocY / cellSize;

                int divRX = (int)Math.Round(divX, MidpointRounding.AwayFromZero);
                int divRY = (int)Math.Round(divY, MidpointRounding.AwayFromZero);

                cellX = (int)(divRX * cellSize);
                cellY = (int)(divRY * cellSize);

                // Interleave the x/y coordinates to create a morton number; use this for cell UID. 
                // This is done because hashing an integer is extremely fast.
                var cellUID = MortonNumber(cellX, cellY);

                if (!meshDict.ContainsKey(cellUID))
                {
                    var newCell = new MeshCell();

                    newCell.LocX = cellX;
                    newCell.LocY = cellY;
                    newCell.Size = cellSize;

                    newCell.Mass += body.Mass;
                    newCell.CmX += body.Mass * body.LocX;
                    newCell.CmY += body.Mass * body.LocY;
                    newCell.BodCount = 1;
                    newCell.ID = cellIdx;

                    meshDict.Add(cellUID, newCell);
                    bodies[b].MeshID = cellIdx;

                    cellIdx++;
                }
                else
                {
                    var cell = meshDict[cellUID];

                    cell.Mass += body.Mass;
                    cell.CmX += body.Mass * body.LocX;
                    cell.CmY += body.Mass * body.LocY;
                    cell.BodCount++;

                    meshDict[cellUID] = cell;
                    bodies[b].MeshID = cell.ID;
                }
            }

            var meshArr = meshDict.Values.ToArray();

            // Calculate the final center of mass for each cell.
            for (int m = 0; m < meshArr.Length; m++)
            {
                meshArr[m].CmX = meshArr[m].CmX / (float)meshArr[m].Mass;
                meshArr[m].CmY = meshArr[m].CmY / (float)meshArr[m].Mass;
            }

            // Build the 2D mesh-body index.
            BuildMeshBodyIndex(meshArr, bodies);

            BuildMeshNeighborIndex(ref meshArr, meshDict, cellSize);

            return meshArr;
        }

        private void BuildMeshBodyIndex(MeshCell[] mesh, Body[] bodies)
        {
            int maxCount = mesh.Max(m => m.BodCount);
            int[,] meshBodIndex = new int[mesh.Length, maxCount];
            // Since the body index array must be a fixed length,
            // we need to track the current element index for each mesh.
            int[] curIdxCount = new int[mesh.Length]; // Number of body indexes added to each mesh.

            for (int b = 0; b < bodies.Length; b++)
            {
                meshBodIndex[bodies[b].MeshID, curIdxCount[bodies[b].MeshID]] = b;
                curIdxCount[bodies[b].MeshID]++;
            }

            _meshBodies = meshBodIndex;
        }

        private void BuildMeshNeighborIndex(ref MeshCell[] mesh, Dictionary<int, MeshCell> meshDict, float cellSize)
        {
            var neighborIdx = new int[mesh.Length, 9];

            for (int i = 0; i < mesh.Length; i++)
            {
                int count = 0;

                for (int x = -1; x <= 1; x++)
                {
                    for (int y = -1; y <= 1; y++)
                    {
                        int nX = (int)(mesh[i].LocX + (cellSize * x));
                        int nY = (int)(mesh[i].LocY + (cellSize * y));

                        var cellUID = MortonNumber(nX, nY);

                        if (meshDict.ContainsKey(cellUID))
                        {
                            neighborIdx[i, count] = meshDict[cellUID].ID;
                            count++;
                        }
                    }
                }

                mesh[i].Neighbors = count;
            }

            _meshNeighbors = neighborIdx;

        }

        [Cudafy]
        public static void CalcForce(GThread gpThread, Body[] inBodies, Body[] outBodies, MeshCell[] inMesh, int[,] inMeshBods, int[,] meshNeighbors, float dt)
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
            MeshCell bodyCell = inMesh[outBody.MeshID];

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
                MeshCell cell = inMesh[b];

                if (cell.ID != outBody.MeshID && IsNeighbor(cell.ID, meshNeighbors, bodyCell.Neighbors, bodyCell.ID) == -1)
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

            for (int i = 0; i < bodyCell.Neighbors; i++)
            {
                int nId = meshNeighbors[bodyCell.ID, i];
                MeshCell cell = inMesh[nId];

                int mbLen = cell.BodCount;
                for (int mb = 0; mb < mbLen; mb++)
                {
                    int meshBodId = inMeshBods[nId, mb];
                    Body inBody = inBodies[meshBodId];

                    if (inBody.UID != outBody.UID)
                    {
                        distX = inBody.LocX - outBody.LocX;
                        distY = inBody.LocY - outBody.LocY;
                        dist = (distX * distX) + (distY * distY);

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
                        if (distSqrt <= colDist)
                        {
                            outBody.HasCollision = 1;
                        }
                    }
                }
            }

            gpThread.SyncThreads();

            outBody.Pressure = GAS_K * (outBody.Density);

            outBodies[a] = outBody;
        }

        [Cudafy]
        public static int IsNeighbor(int testId, int[,] nIdx, int count, int cellId)
        {
            for (int i = 0; i < count; i++)
            {
                if (testId == nIdx[cellId, i])
                    return 0;
            }

            return -1;
        }

        [Cudafy]
        public static void CalcCollisions(GThread gpThread, Body[] inBodies, Body[] outBodies, MeshCell[] inMesh, int[,] inMeshBods, int[,] meshNeighbors, float dt, float viscosity, int drift)
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
                MeshCell bodyCell = inMesh[outBody.MeshID];

                for (int i = 0; i < bodyCell.Neighbors; i++)
                {
                    int nId = meshNeighbors[bodyCell.ID, i];
                    MeshCell cell = inMesh[nId];

                    int mbLen = cell.BodCount;
                    for (int mb = 0; mb < mbLen; mb++)
                    {
                        int meshBodId = inMeshBods[nId, mb];
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
                                    float FLOAT_EPSILON = 1.192092896e-07f;
                                    float FLOAT_EPSILONSQRT = 3.45267E-11f;
                                    float m_kernelSize = 1.0f;

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