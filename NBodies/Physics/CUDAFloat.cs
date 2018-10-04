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
                throw new Exception("Module file not found!  Path: " + modulePath);

                //CudafyTranslator.Language = eLanguage.OpenCL;
                //cudaModule = CudafyTranslator.Cudafy(new Type[] { typeof(Body), typeof(CUDAFloat) });
                //cudaModule.Serialize(modulePath);
            }

            gpu = CudafyHost.GetDevice(eGPUType.OpenCL, gpuIndex);
            gpu.LoadModule(cudaModule);

            Console.WriteLine(gpu.GetDeviceProperties().ToString());
        }

        public void CalcMovement(ref Body[] bodies, float timestep)
        {
            bool altCalcPath = true;

            var blocks = (bodies.Length - 1 + threadsPerBlock - 1) / threadsPerBlock;

            // Zero pad the body array to fit the calculated number of blocks:
            // This makes sure that the dataset fills each block completely,
            // otherwise we run into problems when a block encounters a dataset
            // that doesn't have a work item for each thread.
            int padding = (blocks * threadsPerBlock) - bodies.Length;
            Array.Resize<Body>(ref bodies, (blocks * threadsPerBlock));

            var gpuInBodies = gpu.Allocate(bodies);
            var outBodies = new Body[bodies.Length];
            var gpuOutBodies = gpu.Allocate(outBodies);

            gpu.CopyToDevice(bodies, gpuInBodies);
            gpu.Launch(blocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, timestep);
            gpu.Synchronize();

            // The alternate path skips a bunch of reallocations and memory dumps
            // and just flip-flops the In and Out pointers and launches the collision kernel.
            // I'm not sure if the alt. path is completely stable, but it's definitely faster...
            if (!altCalcPath)
            {
                gpu.CopyFromDevice(gpuOutBodies, bodies);
                gpu.FreeAll();
                gpuInBodies = gpu.Allocate(bodies);
                outBodies = new Body[bodies.Length];
                gpuOutBodies = gpu.Allocate(outBodies);
                gpu.CopyToDevice(bodies, gpuInBodies);
                gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuInBodies, gpuOutBodies, timestep);
                gpu.Synchronize();
                gpu.CopyFromDevice(gpuOutBodies, bodies);
            }
            else
            {
                gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, timestep);
                gpu.Synchronize();
                gpu.CopyFromDevice(gpuInBodies, bodies);
            }

            gpu.FreeAll();
        }

        [Cudafy]
        public static void CalcForce(GThread gpThread, Body[] inBodies, Body[] outBodies, float dt)
        {
            int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            float totMass;
            float force;
            float distX;
            float distY;
            float distSqrt;
            float epsilon = 2;

            Body outBody = inBodies[a];

            if (outBody.Visible == 1)
            {

                outBody.ForceX = 0;
                outBody.ForceY = 0;
                outBody.ForceTot = 0;

                for (int b = 0; b < inBodies.Length; b++)
                {
                    Body inBody = inBodies[b];

                    if (a != b)
                    {
                        if (inBody.Visible == 1)
                        {
                            distX = inBody.LocX - outBody.LocX;
                            distY = inBody.LocY - outBody.LocY;
                            distSqrt = (float)Math.Sqrt(((distX * distX) + (distY * distY)));

                            if (distSqrt > 0f)
                            {
                                totMass = inBody.Mass * outBody.Mass;
                                force = totMass / (distSqrt * distSqrt + epsilon * epsilon);

                                outBody.ForceTot += force;
                                outBody.ForceX += force * distX / distSqrt;
                                outBody.ForceY += force * distY / distSqrt;
                            }
                        }
                    }
                }

                if (outBody.ForceTot > outBody.Mass * 4 & outBody.BlackHole == 0)
                {
                    outBody.InRoche = 1;
                }
                else if (outBody.ForceTot * 2 < outBody.Mass * 4)
                {
                    outBody.InRoche = 0;
                }
                else if (outBody.BlackHole == 2)
                {
                    outBody.InRoche = 1;
                }

            }

            gpThread.SyncThreads();

            outBodies[a] = outBody;

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

            int Master = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            Body outBody = inBodies[Master];

            if (outBody.Visible == 1)
            {
                for (int Slave = 0; Slave < inBodies.Length; Slave++)
                {
                    Body inBody = inBodies[Slave];

                    if (Master != Slave)
                    {
                        if (inBody.Visible == 1)
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
                                        float eps = 1.02f;
                                        int multi = 40;
                                        float friction = 0.2f;

                                        TotMass = M1 * M2;
                                        Force = TotMass / (DistSqrt * DistSqrt + eps * eps);
                                        ForceX = Force * DistX / DistSqrt;
                                        ForceY = Force * DistY / DistSqrt;

                                        outBody.ForceX -= ForceX * multi;
                                        outBody.ForceY -= ForceY * multi;

                                        outBody.SpeedX += dV * vecX * friction;
                                        outBody.SpeedY += dV * vecY * friction;
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
                    }
                }

                // Integrate forces and speeds.
                outBody.SpeedX += dt * outBody.ForceX / outBody.Mass;
                outBody.SpeedY += dt * outBody.ForceY / outBody.Mass;
                outBody.LocX += dt * outBody.SpeedX;
                outBody.LocY += dt * outBody.SpeedY;

            }


            gpThread.SyncThreads();

            outBodies[Master] = outBody;
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