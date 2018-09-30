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
        public static int TargetFPS = 60;
        public static double TimeStep = 0.03f;
        public static int MinFrameTime = 0;

        private static ManualResetEvent _pausePhysics = new ManualResetEvent(true);
        private static bool _skipPhysics = false;
        private static Task _loopTask;
        private static CancellationTokenSource _cancelTokenSource;
        private static Stopwatch _fpsTimer = new Stopwatch();

        public static void StartLoop()
        {
            MinFrameTime = 1000 / TargetFPS;

            _cancelTokenSource = new CancellationTokenSource();

            _loopTask = new Task(DoLoop, _cancelTokenSource.Token, TaskCreationOptions.LongRunning);
            _loopTask.Start();
        }

        public static void Stop()
        {
            _cancelTokenSource.Cancel();
        }

        /// <summary>
        /// Pause physics calculations. Blocks calling thread until any current physics operation is complete.
        /// </summary>
        public static void Pause()
        {
            // Reset the wait handle.
            _pausePhysics.Reset();

            // Wait until the handle is signaled after the GPU calcs complete.
            _pausePhysics.WaitOne(2000);
        }

        /// <summary>
        /// Resume physics calulations.
        /// </summary>
        public static void Resume()
        {
            // Make sure the wait handle has been set 
            // and set the skip bool to false to allow physics to be calculated again.
            _pausePhysics.Set();
            _skipPhysics = false;
        }

        private static void DoLoop()
        {
            while (!_cancelTokenSource.IsCancellationRequested)
            {
                if (!_skipPhysics)
                {
                    if (BodyManager.Bodies.Length > 2)
                    {
                        // CUDA calc.
                        CUDAMain.CalcFrame(BodyManager.Bodies, TimeStep);
                    }
                }

                // If the wait handle is nonsignaled, a pause has been requested.
                if (!_pausePhysics.WaitOne(0))
                {
                    // Set the skip flag then set the wait handle.
                    // This allows the thread which originally called the pause to continue.
                    _skipPhysics = true;
                    _pausePhysics.Set();
                }

                // Draw all the bodies.
                Renderer.DrawBodies(BodyManager.Bodies);

                // FPS Limiter
                DelayFrame();
            }
        }

        private static void DelayFrame()
        {
            if (_fpsTimer.IsRunning)
            {
                long elapTime = _fpsTimer.ElapsedMilliseconds;
                _fpsTimer.Reset();

                if (elapTime >= MinFrameTime)
                {
                    return;
                }
                else
                {
                    var waitTime = (int)(MinFrameTime - elapTime);
                    Thread.Sleep(waitTime);
                    return;
                }
            }
            else
            {
                _fpsTimer.Start();
                return;
            }
        }
    }
}
