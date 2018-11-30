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
        private MeshPoint[] _mesh;
        private MeshPoint[] _rawMesh;
        private int[,] _meshBodies = new int[0, 0];

        public MeshPoint[] CurrentMesh
        {
            get
            {
                return _mesh;
            }
        }

        public MeshPoint[] RawMesh
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
                cudaModule = CudafyTranslator.Cudafy(new Type[] { typeof(Body), typeof(MeshPoint), typeof(CUDAFloat) });
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
                int idx = newcode.IndexOf("Body", lastIdx);

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
                int idx = newcode.IndexOf("MeshPoint", lastIdx);

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
            float viscosity = 20.0f;//40.0f;//5.0f;//7.5f;

            int blocks = 0;

            gpu.StartTimer();

            _mesh = GetNewMesh(bodies);

            var meshCopy = new MeshPoint[_mesh.Length];
            Array.Copy(_mesh, meshCopy, _mesh.Length);

            _rawMesh = meshCopy;

            blocks = BlockCount(_mesh.Length);

            var gpuInBodiesMesh = gpu.Allocate(bodies);
            var gpuInMesh = gpu.Allocate(_mesh);
            var gpuOutMesh = gpu.Allocate(_mesh);
            int[] meshBods = Enumerable.Repeat(-1, bodies.Length).ToArray();
            var gpuOutMeshBods = gpu.Allocate(meshBods);

            gpu.CopyToDevice(bodies, gpuInBodiesMesh);
            gpu.CopyToDevice(_mesh, gpuInMesh);
            gpu.CopyToDevice(meshBods, gpuOutMeshBods);

            gpu.Launch(blocks, threadsPerBlock).PopulateMesh(gpuInBodiesMesh, gpuInMesh, gpuOutMesh, gpuOutMeshBods);

            gpu.CopyFromDevice(gpuOutMesh, _mesh);
            gpu.CopyFromDevice(gpuOutMeshBods, meshBods);

            gpu.FreeAll();

            Console.WriteLine("Populate: " + gpu.StopTimer());


            gpu.StartTimer();

            ShrinkMesh(_mesh, meshBods, ref bodies);

            Console.WriteLine("Prep: " + gpu.StopTimer());


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
                    gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, timestep, viscosity, drift);
                }
            }
            else
            {
                gpu.Launch(blocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, gpuMesh, gpuMeshBodies, timestep);
                gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, timestep, viscosity, 3);
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

        private MeshPoint[] GetNewMesh(Body[] bodies)
        {
            float nodeSize = 30f;
            int padding = (int)nodeSize;

            var maxX = bodies.Max(b => b.LocX) + padding;
            var minX = bodies.Min(b => b.LocX);

            var maxY = bodies.Max(b => b.LocY) + padding;
            var minY = bodies.Min(b => b.LocY);

            float nodeRad = 0;
            //int nodes = 20; //100;
            int nodesPerRow = 0; //100;

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

            nodesPerRow = (int)(meshSize / nodeSize);
            nodesPerRow++;

            nodeRad = nodeSize / 2f;

            float curX = minX;
            float curY = minY;

            MeshPoint[] mesh = new MeshPoint[nodesPerRow * nodesPerRow];

            int rightSteps = 0;
            int downSteps = -1;

            for (int i = 0; i < nodesPerRow * nodesPerRow; i++)
            {
                mesh[i].LocX = curX;
                mesh[i].LocY = curY;
                mesh[i].Mass = 0f;
                mesh[i].Count = 0;
                mesh[i].Size = nodeSize;

                mesh[i].Top = mesh[i].LocY + nodeRad;
                mesh[i].Bottom = mesh[i].LocY - nodeRad;
                mesh[i].Left = mesh[i].LocX - nodeRad;
                mesh[i].Right = mesh[i].LocX + nodeRad;

                curX += nodeSize;
                rightSteps++;

                if (rightSteps == nodesPerRow)
                {
                    curX = minX;
                    rightSteps = 0;

                    curY += nodeSize;
                    downSteps++;
                }

            }

            return mesh;
        }

        private void ShrinkMesh(MeshPoint[] meshes, int[] meshBods, ref Body[] bodies)
        {
            var meshList = new List<MeshPoint>();
            var meshBodDict = new Dictionary<int, int[]>();
            var meshBodList = new List<int[]>();
            int maxCount = meshes.Max(m => m.Count);
            int[] curIdx = new int[meshes.Length];

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
                    var idx = curIdx[meshBods[b]];
                    meshBodDict[meshBods[b]][idx] = b;
                    curIdx[meshBods[b]]++;
                }
            }

            for (int m = 0; m < meshes.Length; m++)
            {
                // Filter out empty meshes.
                if (meshes[m].Count > 0)
                {
                    meshList.Add(meshes[m]);
                    meshBodList.Add(meshBodDict[m]);
                }
            }

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

            _mesh = meshList.ToArray();
            _meshBodies = CreateRectangularArray(meshBodList, maxCount);
        }

        private int[,] CreateRectangularArray(IList<int[]> arrays, int maxLen)
        {
            int minorLength = arrays[0].Length;

            int[,] ret = new int[arrays.Count, maxLen];

            for (int i = 0; i < arrays.Count; i++)
            {
                List<int> bods = new List<int>();
                int idx = 0;
                var array = arrays[i];
                if (array.Length != minorLength)
                {
                    throw new ArgumentException
                        ("All arrays must be the same length");
                }

                for (int j = 0; j < minorLength; j++)
                {
                    if (array[j] != -1)
                    {
                        bods.Add(array[j]);
                    }
                }

                foreach (var bod in bods)
                {
                    ret[i, idx] = bod;
                    idx++;
                }

                for (int x = idx; x < maxLen; x++)
                {
                    ret[i, x] = -1;
                }

            }

            return ret;
        }

        [Cudafy]
        public static void PopulateMesh(GThread gpThread, Body[] inBodies, MeshPoint[] inMeshes, MeshPoint[] outMeshes, int[] outMeshBods)
        {
            int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            if (a > inMeshes.Length)
                return;

            MeshPoint outMesh = inMeshes[a];

            for (int b = 0; b < inBodies.Length; b++)
            {
                Body body = inBodies[b];

                if (body.LocX < outMesh.Right && body.LocX > outMesh.Left && body.LocY < outMesh.Top && body.LocY > outMesh.Bottom)
                {
                    outMesh.Mass += body.Mass;
                    outMesh.CmX += body.Mass * body.LocX;
                    outMesh.CmY += body.Mass * body.LocY;
                    outMesh.Count++;

                    if (outMeshBods[b] == -1)
                    {
                        outMeshBods[b] = a;
                    }
                }

            }

            if (outMesh.Count > 0)
            {
                outMesh.CmX = outMesh.CmX / (float)outMesh.Mass;
                outMesh.CmY = outMesh.CmY / (float)outMesh.Mass;
            }

            gpThread.SyncThreads();

            outMeshes[a] = outMesh;
        }

        [Cudafy]
        public static void CalcForce(GThread gpThread, Body[] inBodies, Body[] outBodies, MeshPoint[] inMesh, int[,] inMeshBods, float dt)
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
                    MeshPoint mesh = inMesh[b];

                    if (mesh.Count > 0)
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

                                        ////// SPH Density Kernel
                                        ////if (outBody.InRoche == 1 && inBody.InRoche == 1)
                                        ////{
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
                                        //}
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
        public static void CalcCollisions(GThread gpThread, Body[] inBodies, Body[] outBodies, float dt, float viscosity, int drift)
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

                int len = inBodies.Length;

                for (int b = 0; b < len; b++)
                {
                    Body inBody = inBodies[b];

                    if (a != b)
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

                                //if (inBody.IsExplosion == 1)
                                //{
                                //    if (outBody.DeltaTime != inBody.DeltaTime)
                                //    {
                                //        outBody.DeltaTime = inBody.DeltaTime;
                                //        outBody.ElapTime = 0.0f;
                                //    }
                                //}


                                ////Shear
                                //float shear = 0.1f;//0.1f;
                                //float normX = DistX / DistSqrt;
                                //float normY = DistY / DistSqrt;

                                //float velo_diffX = inBody.SpeedX - outBody.SpeedX;
                                //float velo_diffY = inBody.SpeedY - outBody.SpeedY;

                                //float tanVelx = velo_diffX - (((velo_diffX - normX) + (velo_diffY * normY)) * normX);
                                //float tanVely = velo_diffY - (((velo_diffX - normX) + (velo_diffY * normY)) * normY);

                                //outBody.ForceX += shear * tanVelx;
                                //outBody.ForceY += shear * tanVely;


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