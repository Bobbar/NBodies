using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Diagnostics;
using System.Drawing;
using NBodies.Physics;
using NBodies.Shapes;

namespace NBodies.Rendering
{
    public static class MainLoop
    {
        public static int TargetFPS = 60;
        public static float TimeStep
        {
            get
            {
                return _timeStep;
            }

            set
            {
                _timeStep = value;
            }
        }
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

        public static Int64 FrameCount
        {
            get
            {
                return _frameCount;
            }
        }

        public static float CurrentFPS = 0;

        private static Int64 _frameCount = 0;
        private static float _timeStep = 0.008f;
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
            try
            {
                while (!_cancelTokenSource.IsCancellationRequested)
                {

                    if (!_skipPhysics)
                    {
                        if (BodyManager.Bodies.Length > 2)
                        {
                            // BodyManager.CullInvisible();

                           // BodyManager.CalcDensityAndPressure();
                            // CUDA calc.
                            PhysicsProvider.PhysicsCalc.CalcMovement(ref BodyManager.Bodies, TimeStep);

                            ProcessRoche(ref BodyManager.Bodies);

                            BodyManager.CullInvisible();

                            _frameCount++;

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
            catch (OperationCanceledException)
            {
                // Fail silently
            }

        }

        private static void DelayFrame()
        {
            int waitTime = 0;

            MinFrameTime = 1000 / TargetFPS;

            if (_fpsTimer.IsRunning)
            {
                long elapTime = _fpsTimer.ElapsedMilliseconds;

                _fpsTimer.Reset();

                if (elapTime <= MinFrameTime)
                {
                    waitTime = (int)(MinFrameTime - elapTime);
                    Thread.Sleep(waitTime);
                }

                CurrentFPS = 1000 / (float)(elapTime + waitTime);
            }
            else
            {
                _fpsTimer.Start();
                return;
            }
        }

        private static void ProcessRoche(ref Body[] bodies)
        {
            int len = bodies.Length;
            for (int b = 0; b < len; b++)
            {
                var body = bodies[b];
                if (bodies[b].Visible == 1 && bodies[b].InRoche == 1)
                {
                    if (bodies[b].Size > 1)
                    {
                        bodies[b].Visible = 0;

                        if (float.IsNaN(bodies[b].LocX))
                            Debugger.Break();

                        FractureBody(bodies[b]);
                    }
                }
            }
        }

        public static void FractureBody(Body body)
        {
            float newSize;
            float newMass;
            int divisor;
            float prevMass;
            float area;

            area = (float)(Math.PI * Math.Pow(body.Size / 2f, 2));
            divisor = (int)area;

            if (divisor <= 1) divisor = 2;
            prevMass = body.Mass;
            area = area / divisor;
            newSize = 1;//BodyManager.CalcRadius(area);
            newMass = prevMass / divisor;

            var ellipse = new Ellipse(new PointF(body.LocX, body.LocY), body.Size * 0.5f);

            for (int f = 0; f < divisor; f++)
            {
                float fLocX = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                float fLocY = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);

                int its = 0;

                while (!PointHelper.PointInsideCircle(ellipse.Location, ellipse.Size, new PointF(fLocX, fLocY)))
                {
                    if (its > 100)
                    {
                        ellipse.Size += 0.5f;
                        its = 0;
                    }

                    fLocX = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                    fLocY = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);

                    its++;
                }

                BodyManager.Add(fLocX, fLocY, body.SpeedX, body.SpeedY, newSize, newMass, Color.FromArgb(body.Color), 1);

            }
        }
    }
}
