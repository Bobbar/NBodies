using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NBodies.CUDA;
using System.Threading;
using System.Diagnostics;

namespace NBodies.Rendering
{
    public static class MainLoop
    {
        public static bool FreezeTime = false;
        public static int TargetFPS = 40;
        public static double TimeStep = 0.05f;
        public static bool Busy = false;

        public static ManualResetEvent UIWindowOpen = new ManualResetEvent(false);

        private static Task loopTask;
        private static CancellationTokenSource cancelTokenSource;
        private static Stopwatch fpsTimer = new Stopwatch();
        private static int minFrameTime = 0;

        public static void StartLoop()
        {
            minFrameTime = 1000 / TargetFPS;

            


            cancelTokenSource = new CancellationTokenSource();

            loopTask = new Task(DoLoop, cancelTokenSource.Token, TaskCreationOptions.LongRunning);
            loopTask.Start();
        }

        public static void Stop()
        {
            cancelTokenSource.Cancel();

        }

        private static void DoLoop()
        {
            while (!cancelTokenSource.IsCancellationRequested)
            {
                
                if (!FreezeTime && BodyManager.Bodies.Length > 2)
                {
                    Busy = true;
                    UIWindowOpen.Reset();
                    // CUDA calc.
                    var bds = CUDAMain.CalcFrame(BodyManager.Bodies, TimeStep);


                    // Process addition bodies. (Roche fracturing)


                    BodyManager.Bodies = bds;
                }


                // Render bodies.
                Renderer.DrawBodies(BodyManager.Bodies);


                // Process UI.

                Busy = false;
                UIWindowOpen.Set();

                DelayFrame();
            }
        }


        private static void DelayFrame()
        {
            if (fpsTimer.IsRunning)
            {
                long elapTime = fpsTimer.ElapsedMilliseconds;
                fpsTimer.Reset();

                if (elapTime >= minFrameTime)
                {
                    return;
                }
                else
                {
                    var waitTime = (int)(minFrameTime - elapTime);
                    Thread.Sleep(waitTime);
                    return;
                }
            }
            else
            {
                fpsTimer.Start();
                return;
            }
        }



    }
}
