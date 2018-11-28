using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Diagnostics;
using NBodies.Rendering;

namespace NBodies.Physics
{
    public class CUDAFloat : IPhysicsCalc
    {
        private int gpuIndex = 2;
        private readonly int threadsPerBlock = 256;
        private GPGPU gpu;

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
                cudaModule = CudafyTranslator.Cudafy(new Type[] { typeof(Body), typeof(CUDAFloat) });
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

            return newcode;
        }

        public void CalcMovement(ref Body[] bodies, float timestep)
        {
            float viscosity = 20.0f;//40.0f;//5.0f;//7.5f;

            var blocks = (int)Math.Round((bodies.Length - 1 + threadsPerBlock - 1) / (float)threadsPerBlock, 0);

            if (((threadsPerBlock * blocks) - bodies.Length) > threadsPerBlock)
            {
                blocks -= 1;
            }
            else if ((threadsPerBlock * blocks) < bodies.Length)
            {
                blocks += 1;
            }

            var gpuInBodies = gpu.Allocate(bodies);
            var gpuOutBodies = gpu.Allocate(bodies);

            gpu.CopyToDevice(bodies, gpuInBodies);

            if (MainLoop.LeapFrog)
            {
                for (int drift = 1; drift > -1; drift--)
                {
                    gpu.Launch(blocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, timestep);
                    gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, timestep, viscosity, drift);
                }
            }
            else
            {

                gpu.StartTimer();
                gpu.Launch(blocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, timestep);
                Console.WriteLine(gpu.StopTimer());

                gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, timestep, viscosity, 3);
            }

            gpu.CopyFromDevice(gpuInBodies, bodies);

            gpu.FreeAll();
        }

        [Cudafy]
        public static void CalcForce(GThread gpThread, Body[] inBodies, Body[] outBodies, float dt)
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

            Body body = inBodies[a];

            body.ForceTot = 0;
            body.ForceX = 0;
            body.ForceY = 0;
            body.HasCollision = 0;

            body.Density = 0;
            body.Pressure = 0;

            ksize = 1.0f;
            ksizeSq = 1.0f;
            factor = 1.566682f;

            // Calculate initial body density.
            fac = 1.566681f;
            body.Density = (body.Mass * fac);

            int len = inBodies.Length;

            for (int b = 0; b < len; b++)
            {
                Body iBody = inBodies[b];

                if (a != b)
                {
                    distX = iBody.LocX - body.LocX;
                    distY = iBody.LocY - body.LocY;
                    dist = (distX * distX) + (distY * distY);

                    if (dist < 0.04f)
                    {
                        dist = 0.04f;
                    }

                    distSqrt = (float)Math.Sqrt(dist);

                    totMass = iBody.Mass * body.Mass;
                    force = totMass / dist;

                    body.ForceTot += force;
                    body.ForceX += (force * distX / distSqrt);
                    body.ForceY += (force * distY / distSqrt);

                    // SPH Density Kernel
                    if (body.InRoche == 1 && iBody.InRoche == 1)
                    {
                        // is this distance close enough for kernal / neighbor calcs ?
                        if (dist <= ksize)
                        {
                            if (dist < FLOAT_EPSILON)
                            {
                                dist = FLOAT_EPSILON;
                            }

                            //  It's a neighbor; accumulate density.
                            diff = ksizeSq - dist;
                            fac = factor * diff * diff * diff;
                            body.Density += body.Mass * fac;
                        }
                    }


                    // Check if the body is within collision distance and set a flag.
                    // This is checked in the collision kernel, and bodies that don't have
                    // the flag set are skipped. This give a huge performance boost in most situations.
                    if (distSqrt <= (body.Size) + (iBody.Size))
                    {
                        body.HasCollision = 1;
                    }
                } // uid != uid
            } // for b 

            gpThread.SyncThreads();

            body.Pressure = GAS_K * (body.Density);

            outBodies[a] = body;
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