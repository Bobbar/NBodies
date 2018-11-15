using NBodies.Physics;
using NBodies.Shapes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using NBodies.IO;

namespace NBodies.Rendering
{
    public static class MainLoop
    {
        public static bool DrawBodies = true;

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

        public static bool Recording
        {
            get
            {
                return _recorder.RecordingActive;
            }
        }

        public static IRecording Recorder
        {
            get
            {
                return _recorder;
            }
        }

        public static float CurrentFPS = 0;

        private static int _minFrameTime = 0;
        private static Int64 _frameCount = 0;
        private static float _timeStep = 0.008f;
        private static ManualResetEventSlim _pausePhysicsWait = new ManualResetEventSlim(true);
        private static ManualResetEvent _stopLoopWait = new ManualResetEvent(true);
        private static ManualResetEventSlim _drawingDoneWait = new ManualResetEventSlim(true);

        private static bool _skipPhysics = false;
        private static Task _loopTask;
        private static CancellationTokenSource _cancelTokenSource;
        private static Stopwatch _fpsTimer = new Stopwatch();

        private static IRecording _recorder = new IO.ProtoBufRecorder();

        public static void StartLoop()
        {
            _minFrameTime = 1000 / TargetFPS;

            _cancelTokenSource = new CancellationTokenSource();

            _loopTask = new Task(DoLoop, _cancelTokenSource.Token, TaskCreationOptions.LongRunning);
            _loopTask.Start();
        }

        public static void Stop()
        {
            _cancelTokenSource.Cancel();
            _stopLoopWait.Reset();

            _stopLoopWait.WaitOne(2000);
        }

        /// <summary>
        /// Pause physics calculations and blocks calling thread until any current physics operation is complete.
        /// </summary>
        public static void WaitForPause()
        {
            // Reset the wait handle.
            _pausePhysicsWait.Reset();
            // Wait until the handle is signaled after the GPU calcs complete.
            _pausePhysicsWait.Wait(2000);
        }

        /// <summary>
        /// Resume physics calulations.
        /// </summary>
        public static void Resume()
        {
            // Make sure the wait handle has been set
            // and set the skip bool to false to allow physics to be calculated again.

            if (!_recorder.PlaybackActive)
            {
                _pausePhysicsWait.Set();
                _skipPhysics = false;
            }
        }

        private static Stopwatch timer = new Stopwatch();


        private async static void DoLoop()
        {
            // 1. Make a copy of the body data and pass that to the physics methods to calculate the next frame.
            // 2. Wait for the drawing thread to finish rendering the previous frame.
            // 3. Once the drawing thread is finished, copy the new data to the current Bodies buffer.
            // 4. Asyncronously start drawing the current Bodies to the field, and immediately loop to start the next physics calc.
            //
            // This allows the rendering and the physics calcs to work at the same time. 
            // This way we can keep the physics thread busy with the next frame while the rendering thread is busy with the current one.
            // The net result is a higher frame rate.
            try
            {
                while (!_cancelTokenSource.IsCancellationRequested)
                {

                    if (!_skipPhysics && !_recorder.PlaybackActive)
                    {
                        if (BodyManager.Bodies.Length > 2)
                        {
                            // Reset all bodies elapsed times if all of them are ready.
                            BodyManager.CheckSetForNextDT();

                            // 1.
                            // Copy the current bodies to another array.
                            var bodiesCopy = new Body[BodyManager.Bodies.Length];
                            Array.Copy(BodyManager.Bodies, bodiesCopy, BodyManager.Bodies.Length);

                            // Calc all physics and movements.
                            PhysicsProvider.PhysicsCalc.CalcMovement(ref bodiesCopy, TimeStep);

                            // 2.
                            // Wait for the drawing thread to complete.
                            _drawingDoneWait.Wait(-1);

                            // 3.
                            // Copy the new data to the current body collection.
                            BodyManager.Bodies = bodiesCopy;

                            // Process and fracture new roche bodies.
                            ProcessRoche(ref BodyManager.Bodies);

                            // Remove invisible bodies.
                            BodyManager.CullInvisible();

                            // Increment physics frame count.
                            _frameCount++;

                            // Send the data to the recorder if we are recording.
                            if (_recorder.RecordingActive && BodyManager.Bodies.Length > 0)
                                _recorder.RecordFrame(BodyManager.Bodies);

                        }
                    }

                    // If the wait handle is nonsignaled, a pause has been requested.
                    if (!_pausePhysicsWait.Wait(0))
                    {
                        // Set the skip flag then set the wait handle.
                        // This allows the thread which originally called the pause to continue.
                        _skipPhysics = true;
                        _pausePhysicsWait.Set();
                    }

                    // Make sure the drawing thread is finished.
                    _drawingDoneWait.Wait(-1);

                    // If we are playing back a recording, get the current field frame
                    // from the recorder and bring it in to be rendered.
                    if (_recorder.PlaybackActive && !_recorder.PlaybackComplete)
                    {
                        var frame = _recorder.GetNextFrame();

                        if (frame != null && frame.Length > 0)
                        {
                            BodyManager.Bodies = frame;
                            BodyManager.RebuildUIDIndex();
                        }
                    }

                    // 4.
                    if (DrawBodies)
                        Renderer.DrawBodiesAsync(BodyManager.Bodies, _drawingDoneWait);


                    // FPS Limiter
                    DelayFrame();
                }
            }
            catch (OperationCanceledException)
            {
                // Fail silently
            }

            if (_cancelTokenSource.IsCancellationRequested)
            {
                _stopLoopWait.Set();
            }
        }

        public static void StartRecording(string file)
        {
            _recorder.StopAll();

            _recorder.CreateRecording(file);
        }

        public static void StopRecording()
        {
            _recorder.StopAll();
        }

        public static IRecording StartPlayback(string file)
        {
            _skipPhysics = true;

            Stop();

            _recorder.OpenRecording(file);

            StartLoop();

            return _recorder;
        }

        public static void StopPlayback()
        {
            _recorder.StopAll();
        }

        private static void DelayFrame()
        {
            int waitTime = 0;

            _minFrameTime = 1000 / TargetFPS;

            if (_fpsTimer.IsRunning)
            {
                long elapTime = _fpsTimer.ElapsedMilliseconds;

                _fpsTimer.Reset();

                if (elapTime <= _minFrameTime)
                {
                    waitTime = (int)(_minFrameTime - elapTime);
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
                if (bodies[b].Visible == 1 && bodies[b].InRoche == 1 && bodies[b].BlackHole != 2 && bodies[b].BlackHole != 1 && bodies[b].IsExplosion != 1)
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
            float minSize = 1.0f;
            float newMass;
            float prevMass;

            bool flipflop = true;

            float density = body.Mass / (float)(Math.PI * Math.Pow(body.Size / 2, 2));

            newMass = BodyManager.CalcMass(1, density);

            //int num = (int)(body.Mass / nMass);

            prevMass = body.Mass;

            var ellipse = new Ellipse(new PointF(body.LocX, body.LocY), body.Size * 0.5f);

            bool done = false;
            float stepSize = minSize;

            //float startXpos = ellipse.Location.X - (ellipse.Size / 2) + stepSize;
            //float startYpos = ellipse.Location.Y - (ellipse.Size / 2) + stepSize;

            float startXpos = ellipse.Location.X - ellipse.Size;
            float startYpos = ellipse.Location.Y - ellipse.Size;

            float Xpos = startXpos;
            float Ypos = startYpos;

            List<PointF> newPoints = new List<PointF>();

            while (!done)
            {
                var testPoint = new PointF(Xpos, Ypos);

                if (PointHelper.PointInsideCircle(ellipse.Location, ellipse.Size, testPoint))
                {
                    newPoints.Add(testPoint);
                }

                Xpos += stepSize;

                if (Xpos > ellipse.Location.X + (ellipse.Size + minSize))
                {
                    if (flipflop)
                    {
                        Xpos = startXpos + (minSize / 2);
                        flipflop = false;
                    }
                    else
                    {
                        Xpos = startXpos;
                        flipflop = true;
                    }

                    Ypos += stepSize - 0.15f;
                }

                if (Ypos > ellipse.Location.Y + (ellipse.Size) + minSize)
                {
                    done = true;
                }
            }

            // newMass = prevMass / newPoints.Count;

            //  float postMass = newMass * newPoints.Count;

            foreach (var pnt in newPoints)
            {
                BodyManager.Add(pnt.X, pnt.Y, body.SpeedX, body.SpeedY, minSize, newMass, Color.FromArgb(body.Color), 1);
            }
        }
    }
}