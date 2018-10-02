using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;

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
            var cudaModule = CudafyModule.TryDeserialize();

            if (cudaModule == null || !cudaModule.TryVerifyChecksums())
            {
                CudafyTranslator.Language = eLanguage.OpenCL;
                cudaModule = CudafyTranslator.Cudafy(new Type[] { typeof(Body), typeof(CUDAFloat) });
                cudaModule.Serialize();
            }

            gpu = CudafyHost.GetDevice(eGPUType.OpenCL, gpuIndex);
            gpu.LoadModule(cudaModule);

            Console.WriteLine(gpu.GetDeviceProperties().ToString());
        }

        public void CalcMovement(Body[] bodies, float timestep)
        {
            bool altCalcPath = true;

            var blocks = (bodies.Length - 1 + threadsPerBlock - 1) / threadsPerBlock;
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

            if (a <= inBodies.Length - 1)
            {
                outBodies[a] = inBodies[a];

                outBodies[a].ForceX = 0;
                outBodies[a].ForceY = 0;
                outBodies[a].ForceTot = 0;

                for (int b = 0; b < inBodies.Length; b++)
                {
                    if (a != b && inBodies[b].Visible == 1)
                    {
                        distX = inBodies[b].LocX - outBodies[a].LocX;
                        distY = inBodies[b].LocY - outBodies[a].LocY;
                        distSqrt = (float)Math.Sqrt(((distX * distX) + (distY * distY)));

                        if (distSqrt > 0f)
                        {
                            totMass = inBodies[b].Mass * outBodies[a].Mass;
                            force = totMass / (distSqrt * distSqrt + epsilon * epsilon);

                            outBodies[a].ForceTot += force;
                            outBodies[a].ForceX += force * distX / distSqrt;
                            outBodies[a].ForceY += force * distY / distSqrt;
                        }
                    }
                }

                if (outBodies[a].ForceTot > outBodies[a].Mass * 4 & outBodies[a].BlackHole == 0)
                {
                    outBodies[a].InRoche = 1;
                }
                else if (outBodies[a].ForceTot * 2 < outBodies[a].Mass * 4)
                {
                    outBodies[a].InRoche = 0;
                }
                else if (outBodies[a].BlackHole == 2)
                {
                    outBodies[a].InRoche = 1;
                }

                gpThread.SyncThreads();
            }
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
            if (Master <= inBodies.Length - 1 & inBodies[Master].Visible == 1)
            {
                outBodies[Master] = inBodies[Master];
                for (int Slave = 0; Slave <= inBodies.Length - 1; Slave++)
                {
                    if (Master != Slave & inBodies[Slave].Visible == 1)
                    {
                        DistX = inBodies[Slave].LocX - outBodies[Master].LocX;
                        DistY = inBodies[Slave].LocY - outBodies[Master].LocY;
                        Dist = (DistX * DistX) + (DistY * DistY);
                        DistSqrt = (float)Math.Sqrt(Dist);

                        if (DistSqrt <= (outBodies[Master].Size * 0.5f) + (inBodies[Slave].Size * 0.5f))
                        {
                            if (DistSqrt > 0)
                            {
                                V1x = outBodies[Master].SpeedX;
                                V1y = outBodies[Master].SpeedY;
                                V2x = inBodies[Slave].SpeedX;
                                V2y = inBodies[Slave].SpeedY;
                                M1 = outBodies[Master].Mass;
                                M2 = inBodies[Slave].Mass;
                                vecX = DistX * 0.5f;
                                vecY = DistY * 0.5f;
                                vecX = vecX / (DistSqrt * 0.5f); // LenG
                                vecY = vecY / (DistSqrt * 0.5f); // LenG
                                V1 = vecX * V1x + vecY * V1y;
                                V2 = vecX * V2x + vecY * V2y;
                                U1 = (M1 * V1 + M2 * V2 - M2 * (V1 - V2)) / (M1 + M2);
                                dV = U1 - V1;

                                if (outBodies[Master].InRoche == 0 & inBodies[Slave].InRoche == 1)
                                {
                                    if (outBodies[Master].Mass > inBodies[Slave].Mass)
                                    {
                                        CollideBodies(outBodies[Master], inBodies[Slave], dV, vecX, vecY);
                                    }
                                    else if (outBodies[Master].Mass == inBodies[Slave].Mass)
                                    {
                                        if (outBodies[Master].UID > inBodies[Slave].UID)
                                        {
                                            CollideBodies(outBodies[Master], inBodies[Slave], dV, vecX, vecY);
                                        }
                                        else
                                        {
                                            outBodies[Master].Visible = 0;
                                        }
                                    }
                                    else
                                    {
                                        outBodies[Master].Visible = 0;
                                    }
                                }
                                else if (outBodies[Master].InRoche == 0 & inBodies[Slave].InRoche == 0)
                                {
                                    if (outBodies[Master].Mass > inBodies[Slave].Mass)
                                    {
                                        CollideBodies(outBodies[Master], inBodies[Slave], dV, vecX, vecY);
                                    }
                                    else if (outBodies[Master].Mass == inBodies[Slave].Mass)
                                    {
                                        if (outBodies[Master].UID > inBodies[Slave].UID)
                                        {
                                            CollideBodies(outBodies[Master], inBodies[Slave], dV, vecX, vecY);
                                        }
                                        else
                                        {
                                            outBodies[Master].Visible = 0;
                                        }
                                    }
                                    else
                                    {
                                        outBodies[Master].Visible = 0;
                                    }
                                }
                                else if (outBodies[Master].InRoche == 1 & inBodies[Slave].InRoche == 1)
                                {
                                    // Lame Spring force attempt. It's literally a reversed gravity force that's increased with a multiplier.
                                    float eps = 1.02f;
                                    int multi = 40;
                                    float friction = 0.2f;

                                    TotMass = M1 * M2;
                                    Force = TotMass / (DistSqrt * DistSqrt + eps * eps);
                                    ForceX = Force * DistX / DistSqrt;
                                    ForceY = Force * DistY / DistSqrt;

                                    outBodies[Master].ForceX -= ForceX * multi;
                                    outBodies[Master].ForceY -= ForceY * multi;

                                    outBodies[Master].SpeedX += dV * vecX * friction;
                                    outBodies[Master].SpeedY += dV * vecY * friction;
                                }
                                else if (outBodies[Master].InRoche == 1 & inBodies[Slave].InRoche == 0)
                                {
                                    outBodies[Master].Visible = 0;
                                }
                            }
                            else if (outBodies[Master].Mass > inBodies[Slave].Mass)
                            {
                                Area1 = (float)Math.PI * (float)(Math.Pow(outBodies[Master].Size, 2));
                                Area2 = (float)Math.PI * (float)(Math.Pow(inBodies[Slave].Size, 2));
                                Area1 = Area1 + Area2;
                                outBodies[Master].Size = (float)Math.Sqrt(Area1 / Math.PI);
                                outBodies[Master].Mass = outBodies[Master].Mass + inBodies[Slave].Mass;
                            }
                            else
                            {
                                outBodies[Master].Visible = 0;
                            }
                        }
                    }
                }

                // Integrate forces and speeds.
                outBodies[Master].SpeedX += dt * outBodies[Master].ForceX / outBodies[Master].Mass;
                outBodies[Master].SpeedY += dt * outBodies[Master].ForceY / outBodies[Master].Mass;
                outBodies[Master].LocX += dt * outBodies[Master].SpeedX;
                outBodies[Master].LocY += dt * outBodies[Master].SpeedY;
            }

            gpThread.SyncThreads();
        }

        [Cudafy]
        public static void CollideBodies(Body master, Body slave, float dV, float vecX, float vecY)
        {
            master.SpeedX += dV * vecX;
            master.SpeedY += dV * vecY;
            float a1 = (float)Math.PI * (float)(Math.Pow(master.Size, 2));
            float a2 = (float)Math.PI * (float)(Math.Pow(slave.Size, 2));
            float a = a1 + a2;
            master.Size = (float)Math.Sqrt(a / Math.PI);
            master.Mass += slave.Mass;
        }
    }
}