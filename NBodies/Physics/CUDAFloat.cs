using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Diagnostics;

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
            //var modulePath = @"..\..\Kernels\CUDAFloat.cdfy";
            var cudaModule = CudafyModule.TryDeserialize();

            if (cudaModule == null || !cudaModule.TryVerifyChecksums())
            {
                // throw new Exception("Module file not found!  Path: " + modulePath);

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
            float viscosity = 5.0f;//7.5f;

            var blocks = (int)Math.Round((bodies.Length - 1 + threadsPerBlock - 1) / (float)threadsPerBlock, 0);

            // Zero pad the body array to fit the calculated number of blocks:
            // This makes sure that the dataset fills each block completely,
            // otherwise we run into problems when a block encounters a dataset
            // that doesn't have a work item for each thread.


            if (bodies.Length < blocks * threadsPerBlock)
                Array.Resize<Body>(ref bodies, (blocks * threadsPerBlock) + 1);

            var gpuInBodies = gpu.Allocate(bodies);
            var gpuOutBodies = gpu.Allocate(bodies);

            gpu.CopyToDevice(bodies, gpuInBodies);

            gpu.Launch(blocks, threadsPerBlock).CalcForce(gpuInBodies, gpuOutBodies, timestep);
            gpu.Launch(blocks, threadsPerBlock).CalcCollisions(gpuOutBodies, gpuInBodies, timestep, viscosity);

            gpu.CopyFromDevice(gpuInBodies, bodies);

            gpu.FreeAll();
        }

        [Cudafy]
        public static void CalcForce(GThread gpThread, Body[] inBodies, Body[] outBodies, float dt)
        {
            float GAS_K = 0.3f;//0.8f;// 0.1f
            float FLOAT_EPSILON = 1.192092896e-07f;

            float ksize = 0;
            float ksizeSq = 0;
            double kernRad9 = 0;
            double factor = 0;
            float diff = 0;
            double fac = 0;

            float totMass;
            float force;
            float distX;
            float distY;
            float dist;
            float distSqrt;

            int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            Body body = inBodies[a];

            if (body.ElapTime < dt)
            {
                body.ForceTot = 0;
                body.ForceX = 0;
                body.ForceY = 0;
                body.HasCollision = 0;

                body.Density = 0;
                body.Pressure = 0;

                ksize = body.Size;
                ksizeSq = ksize * ksize;
                kernRad9 = Math.Pow((double)ksize, 9.0);
                factor = (float)(315.0 / (64.0 * Math.PI * kernRad9));

                // Calculate initial body density.
                diff = ksizeSq - FLOAT_EPSILON;
                fac = factor * diff * diff * diff;
                body.Density = (float)(body.Mass * fac);

                int len = inBodies.Length;

                for (int b = 0; b < len; b++)
                {
                    Body iBody = inBodies[b];

                    if (iBody.Visible == 1)
                    {
                        distX = iBody.LocX - body.LocX;
                        distY = iBody.LocY - body.LocY;
                        dist = (distX * distX) + (distY * distY);

                        if (dist < 0.10f)
                        {
                            dist = 0.10f;
                        }

                        distSqrt = (float)Math.Sqrt(dist);

                        if (iBody.UID != body.UID)
                        {
                            if (distSqrt > 0f)
                            {
                                //if (iBody.IsExplosion != 1)
                                //{
                                totMass = iBody.Mass * body.Mass;
                                //force = totMass / (distSqrt * distSqrt + 0.2f);
                                //force = totMass / (dist + 0.02f);
                                force = totMass / (dist);

                                body.ForceTot += force;
                                body.ForceX += (force * distX / distSqrt);
                                body.ForceY += (force * distY / distSqrt);
                                //}
                            }

                            //// Check if the body is within collision distance and set a flag.
                            //// This is checked in the collision kernel, and bodies that don't have
                            //// the flag set are skipped. This give a huge performance boost in most situations.
                            ////if (distSqrt <= (body.Size * 0.5f) + (iBody.Size * 0.5f))
                            //if (distSqrt <= (body.Size) + (iBody.Size))
                            //{
                            //    body.HasCollision = 1;
                            //}



                            // SPH Density Kernel
                            if (body.InRoche == 1 && iBody.InRoche == 1 && iBody.BlackHole != 1)
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
                                    body.Density += (float)(body.Mass * fac);
                                }
                            }


                            // Check if the body is within collision distance and set a flag.
                            // This is checked in the collision kernel, and bodies that don't have
                            // the flag set are skipped. This give a huge performance boost in most situations.
                            //if (distSqrt <= (body.Size * 0.5f) + (iBody.Size * 0.5f))
                            if (distSqrt <= (body.Size * 2.0f) + (iBody.Size * 2.0f))
                            {
                                body.HasCollision = 1;
                            }
                        } // uid != uid

                    } // ibody.visible

                } // for b 

            }// elap < dt

            gpThread.SyncThreads();


            if (body.Density > 0)
            {
                body.Pressure = GAS_K * (body.Density);// - DENSITY_OFFSET);
            }


            if (body.ForceTot > body.Mass * 4 & body.BlackHole == 0)
            {
                body.InRoche = 1;
            }
            else if (body.ForceTot * 2 < body.Mass * 4)
            {
                body.InRoche = 0;
            }
            else if (body.BlackHole == 2 || body.IsExplosion == 1)
            {
                body.InRoche = 1;
            }

            outBodies[a] = body;
        }

        [Cudafy]
        public static void CalcCollisions(GThread gpThread, Body[] inBodies, Body[] outBodies, float dt, float viscosity)
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
            float DistX;
            float DistY;
            float Dist;
            float DistSqrt;

            int a = gpThread.blockDim.x * gpThread.blockIdx.x + gpThread.threadIdx.x;

            Body outBody = inBodies[a];

            if (outBody.HasCollision == 1) //&& outBody.ElapTime < dt)
            {

                float FLOAT_EPSILON = 1.192092896e-07f;
                float m_kernelSize = outBody.Size;
                double kernelRad6 = Math.Pow((m_kernelSize / 3.0f), 6);
                float m_factorPress = (float)(15.0f / (Math.PI * kernelRad6));
                float m_kernelSizeSq = m_kernelSize * m_kernelSize;

                float visc_kSize3 = (float)Math.Pow(m_kernelSize, 3);
                float visc_Factor = (float)(15.0 / (2.0f * Math.PI * visc_kSize3));


                int len = inBodies.Length;

                if (outBody.Visible == 1)
                {
                    for (int b = 0; b < len; b++)
                    {
                        Body inBody = inBodies[b];

                        if (inBody.UID != outBody.UID && inBody.Visible == 1 && inBody.HasCollision == 1)
                        {
                            DistX = inBody.LocX - outBody.LocX;
                            DistY = inBody.LocY - outBody.LocY;
                            Dist = (DistX * DistX) + (DistY * DistY);
                            DistSqrt = (float)Math.Sqrt(Dist);

                            if (DistSqrt <= (outBody.Size * 0.5f) + (inBody.Size * 0.5f))
                            {
                                if (DistSqrt > 0)
                                {

                                    if (outBody.InRoche == 1 & inBody.InRoche == 1)
                                    {
                                        if (outBody.Density > 0 && inBody.Density > 0)
                                        {
                                            if (Dist < FLOAT_EPSILON)
                                            {
                                                Dist = FLOAT_EPSILON;
                                                DistSqrt = (float)Math.Sqrt(Dist);
                                            }

                                            float scalar = outBody.Mass * (outBody.Pressure + inBody.Pressure) / (2.0f * outBody.Density);
                                            float gradFactor = -m_factorPress * 3.0f * (m_kernelSize - DistSqrt) * (m_kernelSize - DistSqrt) / DistSqrt;

                                            float gradX = (DistX * gradFactor);
                                            float gradY = (DistY * gradFactor);

                                            gradX = gradX * scalar;
                                            gradY = gradY * scalar;

                                            outBody.ForceX += gradX;
                                            outBody.ForceY += gradY;

                                            // Viscosity

                                            //if (outBody.IsExplosion != 1 && inBody.IsExplosion != 1)
                                            //{
                                            float visc_Laplace = visc_Factor * (6.0f / visc_kSize3) * (m_kernelSize - DistSqrt);
                                            float visc_scalar = outBody.Mass * visc_Laplace * viscosity * 1.0f / outBody.Density;

                                            // TODO: Share this with shear.      vvv
                                            float viscVelo_diffX = outBody.SpeedX - inBody.SpeedX;
                                            float viscVelo_diffY = outBody.SpeedY - inBody.SpeedY;

                                            viscVelo_diffX *= visc_scalar;
                                            viscVelo_diffY *= visc_scalar;

                                            outBody.ForceX -= viscVelo_diffX;
                                            outBody.ForceY -= viscVelo_diffY;
                                            //}

                                            if (inBody.IsExplosion == 1)
                                            {

                                                if (outBody.DeltaTime != inBody.DeltaTime)
                                                {
                                                    outBody.DeltaTime = inBody.DeltaTime;
                                                    outBody.ElapTime = 0.0f;
                                                }

                                            }


                                            ////Shear
                                            //float shear = 0.1f;//0.1f;
                                            //float normX = DistX / DistSqrt;
                                            //float normY = DistY / DistSqrt;

                                            //// TODO: Share this with viscosity.  ^^^
                                            //float velo_diffX = inBody.SpeedX - outBody.SpeedX;
                                            //float velo_diffY = inBody.SpeedY - outBody.SpeedY;

                                            //float tanVelx = velo_diffX - (((velo_diffX - normX) + (velo_diffY * normY)) * normX);
                                            //float tanVely = velo_diffY - (((velo_diffX - normX) + (velo_diffY * normY)) * normY);

                                            //outBody.ForceX += shear * tanVelx;
                                            //outBody.ForceY += shear * tanVely;

                                        }
                                    }
                                    else if (outBody.InRoche == 1 & inBody.InRoche == 0)
                                    {
                                        outBody.Visible = 0;
                                    }
                                    else
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
            }


            // Integrate forces and speeds.
            //outBody.SpeedX += dt * outBody.ForceX / outBody.Mass;
            //outBody.SpeedY += dt * outBody.ForceY / outBody.Mass;
            //outBody.LocX += dt * (outBody.SpeedX * 0.99f);
            //outBody.LocY += dt * (outBody.SpeedY * 0.99f);

            //if (outBody.Lifetime > 0.0f)
            //    outBody.Age += (dt * 4.0f);


            if (outBody.ElapTime < dt)
            {
                outBody.SpeedX += outBody.DeltaTime * outBody.ForceX / outBody.Mass;
                outBody.SpeedY += outBody.DeltaTime * outBody.ForceY / outBody.Mass;
                outBody.LocX += outBody.DeltaTime * (outBody.SpeedX * 0.99f);
                outBody.LocY += outBody.DeltaTime * (outBody.SpeedY * 0.99f);

                if (outBody.Lifetime > 0.0f)
                    outBody.Age += (outBody.DeltaTime * 4.0f);

                outBody.ElapTime += outBody.DeltaTime;
            }


            gpThread.SyncThreads();

            outBodies[a] = outBody;
        }

        [Cudafy]
        public static Body CollideBodies(Body master, Body slave, float dV, float vecX, float vecY)
        {
            Body bodyA = master;
            Body bodyB = slave;

            bodyA.SpeedX += dV * vecX;
            bodyA.SpeedY += dV * vecY;

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