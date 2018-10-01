using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NBodies.CUDA;
using System.Threading;
using System.Diagnostics;
using System.Drawing;

namespace NBodies.Rendering
{
    public static class MainLoop
    {
        public static int TargetFPS = 60;
        public static double TimeStep = 0.008f;
        public static int MinFrameTime = 0;

        public static bool PausePhysics
        {
            get
            {
                return _skipPhysics;
            }

            set
            {
                _skipPhysics = value;
            }
        }

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
        /// Pause physics calculations and blocks calling thread until any current physics operation is complete.
        /// </summary>
        public static void WaitForPause()
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

                        ProcessRoche(ref BodyManager.Bodies);
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
            MinFrameTime = 1000 / TargetFPS;

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

        private static void ProcessRoche(ref CUDAMain.Body[] bodies)
        {
            int len = bodies.Length;
            for (int b = 0; b < len; b++)
            {
                if (bodies[b].Visible == 1 && bodies[b].InRoche == 1)
                {
                    if (bodies[b].Size > 1)
                    {
                        bodies[b].Visible = 0;
                        FractureBody(bodies[b]);
                    }
                }
            }
        }

        public static void FractureBody(CUDAMain.Body body)
        {
            double newSize;
            double newMass;
            int divisor;
            double prevMass;
            double area;

            //if (body.Visible == 1 && body.Size > 1)
            //{
            area = Math.PI * Math.Pow(body.Size / 2f, 2);
            divisor = (int)area;

            if (divisor <= 1) divisor = 2;
            prevMass = body.Mass;
            area = area / (float)divisor;
            newSize = BodyManager.CalcRadius(area);
            newMass = prevMass / (float)divisor;

            for (int f = 0; f < divisor; f++)
            {

                double fLocX = Numbers.GetRandomDouble(body.LocX - body.Size * 0.5f, body.LocX + body.Size * 0.5f);
                double fLocY = Numbers.GetRandomDouble(body.LocY - body.Size * 0.5f, body.LocY + body.Size * 0.5f);

                while (!PointHelper.PointInsideCircle(new PointF().FromDouble(body.LocX, body.LocY), (float)body.Size, new PointF().FromDouble(fLocX, fLocY)))
                {

                    fLocX = Numbers.GetRandomDouble(body.LocX - body.Size * 0.5f, body.LocX + body.Size * 0.5f);
                    fLocY = Numbers.GetRandomDouble(body.LocY - body.Size * 0.5f, body.LocY + body.Size * 0.5f);

                }

                BodyManager.Add(fLocX, fLocY, body.SpeedX, body.SpeedY, newSize, newMass, Color.FromArgb(body.Color), 1);

            }
            //}

        }
    }
}
