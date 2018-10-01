using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using ProtoBuf;

namespace NBodies.CUDA
{
    public static class CUDAMain
    {
        [Cudafy]
        [ProtoContract]
        public struct Body
        {
            [ProtoMember(1)]
            public double LocX;

            [ProtoMember(2)]
            public double LocY;

            [ProtoMember(3)]
            public double Mass;

            [ProtoMember(4)]
            public double SpeedX;

            [ProtoMember(5)]
            public double SpeedY;

            [ProtoMember(6)]
            public double ForceX;

            [ProtoMember(7)]
            public double ForceY;

            [ProtoMember(8)]
            public double ForceTot;

            [ProtoMember(9)]
            public int Color;

            [ProtoMember(10)]
            public double Size;

            [ProtoMember(11)]
            public int Visible;

            [ProtoMember(12)]
            public int InRoche;

            [ProtoMember(13)]
            public int BlackHole;

            [ProtoMember(14)]
            public int UID;


        }

        private static int gpuIndex = 0;
        private static int threadsPerBlock = 256;
        private static GPGPU gpu;

        public static void InitGPU(int gpuIdx)
        {

            gpuIndex = gpuIdx;

            var cudaModule = CudafyModule.TryDeserialize();

            if (cudaModule == null || !cudaModule.TryVerifyChecksums())
            {
                CudafyTranslator.Language = eLanguage.OpenCL;
                cudaModule = CudafyTranslator.Cudafy();
                cudaModule.Serialize();
            }

            gpu = CudafyHost.GetDevice(eGPUType.OpenCL, gpuIndex);
            gpu.LoadModule(cudaModule);

            Console.WriteLine(gpu.GetDeviceProperties().ToString());
        }

        public static void CalcFrame(Body[] bodies, double timestep)
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
        public static void CalcForce(GThread gpThread, Body[] inBodies, Body[] outBodies, double dt)
        {
            int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            double totMass;
            double force;
            double forceX;
            double forceY;
            double distX;
            double distY;
            double dist;
            double distSqrt;
            double m1, m2;
            double epsilon = 2;

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
                        distSqrt = Math.Sqrt(((distX * distX) + (distY * distY)));

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
        public static void CalcCollisions(GThread gpThread, Body[] inBodies, Body[] outBodies, double dt)
        {
            double VeKY;
            double VekX;
            double V1x;
            double V2x;
            double M1;
            double M2;
            double V1y;
            double V2y;

            double V1;
            double V2;
            double U2;
            double U1;
            double PrevSpdX, PrevSpdY;
            double Area1;
            double Area2;
            double TotMass;
            double Force;
            double ForceX;
            double ForceY;
            double DistX;
            double DistY;
            double Dist;
            double DistSqrt;

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
                        DistSqrt = Math.Sqrt(Dist);

                        if (DistSqrt <= (outBodies[Master].Size / (double)2) + (inBodies[Slave].Size / (double)2))
                        {

                            if (DistSqrt > 0)
                            {
                                V1x = outBodies[Master].SpeedX;
                                V1y = outBodies[Master].SpeedY;
                                V2x = inBodies[Slave].SpeedX;
                                V2y = inBodies[Slave].SpeedY;
                                M1 = outBodies[Master].Mass;
                                M2 = inBodies[Slave].Mass;
                                VekX = DistX / 2; // (Ball(A).LocX - Ball(B).LocX) / 2
                                VeKY = DistY / 2; // (Ball(A).LocY - Ball(B).LocY) / 2
                                VekX = VekX / (DistSqrt / 2); // LenG
                                VeKY = VeKY / (DistSqrt / 2); // LenG
                                V1 = VekX * V1x + VeKY * V1y;
                                V2 = VekX * V2x + VeKY * V2y;
                                U1 = (M1 * V1 + M2 * V2 - M2 * (V1 - V2)) / (M1 + M2);
                                U2 = (M1 * V1 + M2 * V2 - M1 * (V2 - V1)) / (M1 + M2);
                                if (outBodies[Master].InRoche == 0 & inBodies[Slave].InRoche == 1)
                                {
                                    if (outBodies[Master].Mass > inBodies[Slave].Mass)
                                    {
                                        PrevSpdX = outBodies[Master].SpeedX;
                                        PrevSpdY = outBodies[Master].SpeedY;
                                        outBodies[Master].SpeedX = outBodies[Master].SpeedX + (U1 - V1) * VekX;
                                        outBodies[Master].SpeedY = outBodies[Master].SpeedY + (U1 - V1) * VeKY;
                                        // inBodies(Slave).Visible = 0
                                        Area1 = Math.PI * (Math.Pow(outBodies[Master].Size, 2));
                                        Area2 = Math.PI * (Math.Pow(inBodies[Slave].Size, 2));
                                        Area1 = Area1 + Area2;
                                        outBodies[Master].Size = Math.Sqrt(Area1 / Math.PI);
                                        outBodies[Master].Mass = outBodies[Master].Mass + inBodies[Slave].Mass; // Sqr(Ball(B).Mass)
                                    }
                                    else if (outBodies[Master].Mass == inBodies[Slave].Mass)
                                    {
                                        if (outBodies[Master].UID > inBodies[Slave].UID)
                                        {
                                            PrevSpdX = outBodies[Master].SpeedX;
                                            PrevSpdY = outBodies[Master].SpeedY;
                                            outBodies[Master].SpeedX = outBodies[Master].SpeedX + (U1 - V1) * VekX;
                                            outBodies[Master].SpeedY = outBodies[Master].SpeedY + (U1 - V1) * VeKY;
                                            // inBodies(Slave).Visible = 0
                                            Area1 = Math.PI * (Math.Pow(outBodies[Master].Size, 2));
                                            Area2 = Math.PI * (Math.Pow(inBodies[Slave].Size, 2));
                                            Area1 = Area1 + Area2;
                                            outBodies[Master].Size = Math.Sqrt(Area1 / Math.PI);
                                            outBodies[Master].Mass = outBodies[Master].Mass + inBodies[Slave].Mass; // Sqr(Ball(B).Mass)
                                        }
                                        else
                                            outBodies[Master].Visible = 0;
                                    }
                                    else
                                        outBodies[Master].Visible = 0;
                                }
                                else if (outBodies[Master].InRoche == 0 & inBodies[Slave].InRoche == 0)
                                {
                                    if (outBodies[Master].Mass > inBodies[Slave].Mass)
                                    {
                                        PrevSpdX = outBodies[Master].SpeedX;
                                        PrevSpdY = outBodies[Master].SpeedY;
                                        outBodies[Master].SpeedX = outBodies[Master].SpeedX + (U1 - V1) * VekX;
                                        outBodies[Master].SpeedY = outBodies[Master].SpeedY + (U1 - V1) * VeKY;
                                        // inBodies(Slave).Visible = 0
                                        Area1 = Math.PI * (Math.Pow(outBodies[Master].Size, 2));
                                        Area2 = Math.PI * (Math.Pow(inBodies[Slave].Size, 2));
                                        Area1 = Area1 + Area2;
                                        outBodies[Master].Size = Math.Sqrt(Area1 / Math.PI);
                                        outBodies[Master].Mass = outBodies[Master].Mass + inBodies[Slave].Mass; // Sqr(Ball(B).Mass)
                                    }
                                    else if (outBodies[Master].Mass == inBodies[Slave].Mass)
                                    {
                                        if (outBodies[Master].UID > inBodies[Slave].UID)
                                        {
                                            PrevSpdX = outBodies[Master].SpeedX;
                                            PrevSpdY = outBodies[Master].SpeedY;
                                            outBodies[Master].SpeedX = outBodies[Master].SpeedX + (U1 - V1) * VekX;
                                            outBodies[Master].SpeedY = outBodies[Master].SpeedY + (U1 - V1) * VeKY;
                                            // inBodies(Slave).Visible = 0
                                            Area1 = Math.PI * (Math.Pow(outBodies[Master].Size, 2));
                                            Area2 = Math.PI * (Math.Pow(inBodies[Slave].Size, 2));
                                            Area1 = Area1 + Area2;
                                            outBodies[Master].Size = Math.Sqrt(Area1 / Math.PI);
                                            outBodies[Master].Mass = outBodies[Master].Mass + inBodies[Slave].Mass; // Sqr(Ball(B).Mass)
                                        }
                                        else
                                            outBodies[Master].Visible = 0;
                                    }
                                    else
                                        outBodies[Master].Visible = 0;
                                }
                                else if (outBodies[Master].InRoche == 1 & inBodies[Slave].InRoche == 1)
                                {

                                    // Lame Spring force attempt. It's literally a reversed gravity force that's increased with a multiplier.
                                    M1 = outBodies[Master].Mass;
                                    M2 = inBodies[Slave].Mass;
                                    TotMass = M1 * M2;
                                    // TotMass = 100
                                    double EPS = 1.02;
                                    Force = TotMass / (DistSqrt * DistSqrt + EPS * EPS);
                                    ForceX = Force * DistX / DistSqrt;
                                    ForceY = Force * DistY / DistSqrt;
                                    int multi = 40;
                                    outBodies[Master].ForceX -= ForceX * multi;
                                    outBodies[Master].ForceY -= ForceY * multi;

                                    double Friction = 0.2;
                                    outBodies[Master].SpeedX += (U1 - V1) * VekX * Friction;
                                    outBodies[Master].SpeedY += (U1 - V1) * VeKY * Friction;
                                }
                                else if (outBodies[Master].InRoche == 1 & inBodies[Slave].InRoche == 0)
                                    outBodies[Master].Visible = 0;
                            }
                            else if (outBodies[Master].Mass > inBodies[Slave].Mass)
                            {
                                Area1 = Math.PI * (Math.Pow(outBodies[Master].Size, 2));
                                Area2 = Math.PI * (Math.Pow(inBodies[Slave].Size, 2));
                                Area1 = Area1 + Area2;
                                outBodies[Master].Size = Math.Sqrt(Area1 / Math.PI);
                                outBodies[Master].Mass = outBodies[Master].Mass + inBodies[Slave].Mass; // Sqr(Ball(B).Mass)
                            }
                            else
                                // Area1 = PI * (outBodies(Master).Size ^ 2)
                                // Area2 = PI * (inBodies(Slave).Size ^ 2)
                                // Area1 = Area1 + Area2
                                // inBodies(Slave).Size = Sqrt(Area1 / PI)
                                // inBodies(Slave).Mass = inBodies(Slave).Mass + outBodies(Master).Mass 'Sqr(Ball(B).Mass)
                                outBodies[Master].Visible = 0;
                        }
                    }
                }

                outBodies[Master].SpeedX += dt * outBodies[Master].ForceX / (double)outBodies[Master].Mass;
                outBodies[Master].SpeedY += dt * outBodies[Master].ForceY / (double)outBodies[Master].Mass;
                outBodies[Master].LocX += dt * outBodies[Master].SpeedX;
                outBodies[Master].LocY += dt * outBodies[Master].SpeedY;
            }


            gpThread.SyncThreads();
        }

    }

}
