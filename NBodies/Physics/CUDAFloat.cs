using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Diagnostics;

namespace NBodies.Physics
{
    public class CUDAFloat : IPhysicsCalc
    {
        private int gpuIndex = 0;
        private int threadsPerBlock = 256;
        private GPGPU gpu;

        public CUDAFloat(int gpuIdx)
        {
            gpuIndex = gpuIdx;
        }

        public void Init()
        {
            var modulePath = @"..\..\Kernels\CUDAFloat.cdfy";
            var cudaModule = CudafyModule.TryDeserialize(modulePath);

            if (cudaModule == null)// || !cudaModule.TryVerifyChecksums())
            {
                // throw new Exception("Module file not found!  Path: " + modulePath);

                CudafyTranslator.Language = eLanguage.OpenCL;
                cudaModule = CudafyTranslator.Cudafy(new Type[] { typeof(Body), typeof(CUDAFloat) });
                cudaModule.Serialize(modulePath);

            }

            gpu = CudafyHost.GetDevice(eGPUType.OpenCL, gpuIndex);
            gpu.LoadModule(cudaModule);

            Console.WriteLine(gpu.GetDeviceProperties().ToString());
        }

        public void CalcMovement(ref Body[] bodies, float timestep)
        {
            bool altCalcPath = true;

            var blocks = (bodies.Length / threadsPerBlock) + 1;//(bodies.Length - 1 + threadsPerBlock - 1) / threadsPerBlock;

            // Zero pad the body array to fit the calculated number of blocks:
            // This makes sure that the dataset fills each block completely,
            // otherwise we run into problems when a block encounters a dataset
            // that doesn't have a work item for each thread.
            if (bodies.Length <= blocks * threadsPerBlock)
                Array.Resize<Body>(ref bodies, (blocks * threadsPerBlock));

            var gpuInBodies = gpu.Allocate(bodies);
            var outBodies = new Body[bodies.Length];
            var gpuOutBodies = gpu.Allocate(outBodies);

            gpu.CopyToDevice(bodies, gpuInBodies);
            gpu.Launch(blocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, timestep);
            gpu.Synchronize();

            //  gpu.CopyFromDevice(gpuOutBodies, bodies);

            //// The alternate path skips a bunch of reallocations and memory dumps
            //// and just flip-flops the In and Out pointers and launches the collision kernel.
            //// I'm not sure if the alt. path is completely stable, but it's definitely faster...
            //if (!altCalcPath)
            //{
            //    gpu.CopyFromDevice(gpuOutBodies, bodies);
            //    gpu.FreeAll();
            //    gpuInBodies = gpu.Allocate(bodies);
            //    outBodies = new Body[bodies.Length];
            //    gpuOutBodies = gpu.Allocate(outBodies);
            //    gpu.CopyToDevice(bodies, gpuInBodies);
            //    gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuInBodies, gpuOutBodies, timestep);
            //    gpu.Synchronize();
            //    gpu.CopyFromDevice(gpuOutBodies, bodies);
            //}
            //else
            //{
            //    gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, timestep);
            //    gpu.Synchronize();
            //    gpu.CopyFromDevice(gpuInBodies, bodies);
            //}

            gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, timestep);
            gpu.Synchronize();
            gpu.CopyFromDevice(gpuInBodies, bodies);


            gpu.FreeAll();
        }

        [Cudafy]
        public static void CalcForce(GThread gpThread, Body[] inBodies, Body[] outBodies, float dt)
        {
            // int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            float totMass;
            float force;
            float distX;
            float distY;
            float distSqrt;
            float epsilon = 2;

            int gti = gpThread.get_global_id(0);
            int ti = gpThread.get_local_id(0);

            int n = gpThread.get_global_size(0);
            int nt = gpThread.get_local_size(0);
            int nb = n / nt;

            Body body = inBodies[gti];
            body.ForceTot = 0;
            body.ForceX = 0;
            body.ForceY = 0;

            var bodyCache = gpThread.AllocateShared<Body>("bodyCache", 512);

            for (int jb = 0; jb < nb; jb++)
            {
                bodyCache[ti] = inBodies[jb * nt + ti];

                gpThread.SyncThreads();

                for (int j = 0; j < nt; j++)
                {
                    Body iBody = bodyCache[j];

                    if (iBody.UID != body.UID)
                    {
                        distX = iBody.LocX - body.LocX;
                        distY = iBody.LocY - body.LocY;
                        distSqrt = (float)Math.Sqrt(((distX * distX) + (distY * distY)));

                        if (distSqrt > 0f)
                        {
                            //totMass = inBodies[b].Mass * outBodies[a].Mass;
                            totMass = iBody.Mass * body.Mass;

                            force = totMass / (distSqrt * distSqrt + epsilon * epsilon);

                            body.ForceTot += force;
                            body.ForceX += force * distX / distSqrt;
                            body.ForceY += force * distY / distSqrt;
                        }
                    }
                }
                gpThread.SyncThreads();

            }

            if (body.ForceTot > body.Mass * 4 & body.BlackHole == 0)
            {
                body.InRoche = 1;
            }
            else if (body.ForceTot * 2 < body.Mass * 4)
            {
                body.InRoche = 0;
            }
            else if (body.BlackHole == 2)
            {
                body.InRoche = 1;
            }

            ////Integrate forces and speeds.
            //body.SpeedX += dt * body.ForceX / body.Mass;
            //body.SpeedY += dt * body.ForceY / body.Mass;
            //body.LocX += dt * body.SpeedX;
            //body.LocY += dt * body.SpeedY;


            outBodies[gti] = body;

            //   gpThread.SyncThreads();

            //outBodies[a] = outBody;

        }

        [Cudafy]
        public static void CalcCollisions(GThread gpThread, Body[] inBodies, Body[] outBodies, float dt)
        {
            float vecY;
            float vecX;
            float V1x;
            float V2x;
            float M1;
            float M2;
            float V1y;
            float V2y;
            float V1;
            float V2;
            float U1;
            float dV;
            float Area1;
            float Area2;
            float TotMass;
            float Force;
            float ForceX;
            float ForceY;
            float DistX;
            float DistY;
            float Dist;
            float DistSqrt;

            int gti = gpThread.get_global_id(0);
            int ti = gpThread.get_local_id(0);

            int n = gpThread.get_global_size(0);
            int nt = gpThread.get_local_size(0);
            int nb = n / nt;

            Body outBody = inBodies[gti];


            var bodyCacheCol = gpThread.AllocateShared<Body>("bodyCacheCol", 512);


            //int Master = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            //Body outBody = inBodies[Master];


            for (int jb = 0; jb < nb; jb++)
            {
                bodyCacheCol[ti] = inBodies[jb * nt + ti];

                gpThread.SyncThreads();


                if (outBody.Visible == 1)
                {

                    for (int j = 0; j < nt; j++)
                    {
                        Body inBody = bodyCacheCol[j];

                        //if (Master != Slave)
                        //{
                        //if (inBody.Visible == 1)
                        //{

                        if (inBody.UID != outBody.UID && inBody.Visible == 1)
                        {
                            DistX = inBody.LocX - outBody.LocX;
                            DistY = inBody.LocY - outBody.LocY;
                            Dist = (DistX * DistX) + (DistY * DistY);
                            DistSqrt = (float)Math.Sqrt(Dist);

                            if (DistSqrt <= (outBody.Size * 0.5f) + (inBody.Size * 0.5f))
                            {
                                if (DistSqrt > 0)
                                {
                                    V1x = outBody.SpeedX;
                                    V1y = outBody.SpeedY;
                                    V2x = inBody.SpeedX;
                                    V2y = inBody.SpeedY;
                                    M1 = outBody.Mass;
                                    M2 = inBody.Mass;
                                    vecX = DistX * 0.5f;
                                    vecY = DistY * 0.5f;
                                    vecX = vecX / (DistSqrt * 0.5f); // LenG
                                    vecY = vecY / (DistSqrt * 0.5f); // LenG
                                    V1 = vecX * V1x + vecY * V1y;
                                    V2 = vecX * V2x + vecY * V2y;
                                    U1 = (M1 * V1 + M2 * V2 - M2 * (V1 - V2)) / (M1 + M2);
                                    dV = U1 - V1;

                                    if (outBody.InRoche == 0 & inBody.InRoche == 1)
                                    {
                                        if (outBody.Mass > inBody.Mass)
                                        {
                                            outBody = CollideBodies(outBody, inBody, dV, vecX, vecY);
                                        }
                                        else if (outBody.Mass == inBody.Mass)
                                        {
                                            if (outBody.UID > inBody.UID)
                                            {
                                                outBody = CollideBodies(outBody, inBody, dV, vecX, vecY);
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
                                    else if (outBody.InRoche == 0 & inBody.InRoche == 0)
                                    {
                                        if (outBody.Mass > inBody.Mass)
                                        {
                                            outBody = CollideBodies(outBody, inBody, dV, vecX, vecY);
                                        }
                                        else if (outBody.Mass == inBody.Mass)
                                        {
                                            if (outBody.UID > inBody.UID)
                                            {
                                                outBody = CollideBodies(outBody, inBody, dV, vecX, vecY);
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
                                    else if (outBody.InRoche == 1 & inBody.InRoche == 1)
                                    {
                                        // Lame Spring force attempt. It's literally a reversed gravity force that's increased with a multiplier.
                                        float eps = 0.1f;
                                        int multi = 40;
                                        float friction = 0.5f;

                                        float m_kernelSize = outBody.Size;
                                        float m_kernelSize3 = m_kernelSize * m_kernelSize * m_kernelSize;
                                        //float DistSq = (float)Math.Sqrt(Dist);

                                        float diff = (outBody.Size - Dist);
                                        float factor = (diff * diff) / Dist;



                                        double kernelRad6 = Math.Pow((m_kernelSize / 3.0f), 6);
                                        float m_factorPress = (float)(15.0f / (Math.PI * kernelRad6));
                                        float m_kernelSizeSq = m_kernelSize * m_kernelSize;

                                        float scalar = inBody.Mass * (outBody.Pressure + inBody.Pressure) / (2.0f * inBody.Density);

                                        float gradFactor = -m_factorPress * 3.0f * (m_kernelSize - DistSqrt) * (m_kernelSize - DistSqrt) / DistSqrt;

                                        float gradX = (DistX * gradFactor);
                                        float gradY = (DistY * gradFactor);

                                        gradX = gradX * scalar;
                                        gradY = gradY * scalar;

                                        outBody.ForceX += gradX;
                                        outBody.ForceY += gradY;









                                        float visc_kSize3 = (float)Math.Pow(outBody.Size, 3);
                                        float visc_Factor = (float)(15.0 / (2.0f * Math.PI * visc_kSize3));
                                        float visc_Laplace = visc_Factor * (6.0f / visc_kSize3) * (outBody.Size - Dist);


                                        //TotMass = M1 * M2;
                                        //// Force = TotMass / (DistSqrt * DistSqrt + eps * eps);
                                        //Force = TotMass / (Dist + eps);
                                        //ForceX = Force * DistX / DistSqrt;
                                        //ForceY = Force * DistY / DistSqrt;

                                        //outBody.ForceX -= ForceX * factor;
                                        //outBody.ForceY -= ForceY * factor;


                                        float visc_scalar = outBody.Mass * visc_Laplace * 0.8f * 1 / 40;
                                        float velo_diffX = inBody.SpeedX - outBody.SpeedX;
                                        float velo_diffY = inBody.SpeedY - outBody.SpeedY;

                                        velo_diffX *= visc_scalar;
                                        velo_diffY *= visc_scalar;

                                        outBody.ForceX += velo_diffX;
                                        outBody.ForceY += velo_diffY;
                                        //outBody.ForceX -= ForceX * multi;
                                        //outBody.ForceY -= ForceY * multi;

                                        //outBody.SpeedX += dV * vecX * friction;
                                        //outBody.SpeedY += dV * vecY * friction;
                                    }
                                    else if (outBody.InRoche == 1 & inBody.InRoche == 0)
                                    {
                                        outBody.Visible = 0;
                                    }
                                }
                                else if (outBody.Mass > inBody.Mass)
                                {
                                    Area1 = (float)Math.PI * (float)(Math.Pow(outBody.Size, 2));
                                    Area2 = (float)Math.PI * (float)(Math.Pow(inBody.Size, 2));
                                    Area1 = Area1 + Area2;
                                    outBody.Size = (float)Math.Sqrt(Area1 / Math.PI);
                                    outBody.Mass = outBody.Mass + inBody.Mass;
                                }
                                else
                                {
                                    outBody.Visible = 0;
                                }
                            }
                        }

                        //}
                        // }
                    }
                }
                gpThread.SyncThreads();

            }

            // Integrate forces and speeds.
            outBody.SpeedX += dt * outBody.ForceX / outBody.Mass;
            outBody.SpeedY += dt * outBody.ForceY / outBody.Mass;
            outBody.LocX += dt * outBody.SpeedX;
            outBody.LocY += dt * outBody.SpeedY;

            //}

            outBodies[gti] = outBody;


            //gpThread.SyncThreads();

            //outBodies[Master] = outBody;
        }

        [Cudafy]
        public static Body CollideBodies(Body master, Body slave, float dV, float vecX, float vecY)
        {
            Body bodyA = master;
            Body bodyB = slave;

            bodyA.SpeedX += dV * vecX;
            bodyA.SpeedY += dV * vecY;
            float a1 = (float)Math.PI * (float)(Math.Pow(bodyA.Size, 2));
            float a2 = (float)Math.PI * (float)(Math.Pow(bodyB.Size, 2));
            float a = a1 + a2;
            bodyA.Size = (float)Math.Sqrt(a / Math.PI);
            bodyA.Mass += bodyB.Mass;

            return bodyA;
        }
    }
}