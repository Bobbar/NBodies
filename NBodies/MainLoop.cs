using NBodies.Extensions;
using NBodies.IO;
using NBodies.Physics;
using NBodies.Shapes;
using NBodies.Rendering;
using NBodies.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;

namespace NBodies
{
    public static class MainLoop
    {
        public static bool DrawBodies = true;
        public static bool RocheLimit = true;
        public static bool Collisions = true;
        public static bool RewindBuffer = false;
        public const int DefaultThreadsPerBlock = 256;
        public static RenderBase Renderer;

        #region Public Properties

        public static float CurrentFPS
        {
            get { return _currentFPS; }
        }

        public static float PeakFPS
        {
            get { return _peakFPS; }
        }

        public static float CullDistance
        {
            get
            {
                return _cullDistance;
            }
        }

        public static int CellSizeExp
        {
            get
            {
                return _cellSizeExp;
            }

            set
            {
                if (value >= 1 && value <= 8)
                {
                    _cellSizeExp = value;
                }
            }
        }

        public static int MeshLevels
        {
            get
            {
                return _meshLevels;
            }

            set
            {
                if (value >= 1 & value <= 10)
                {
                    _meshLevels = value;
                }
            }
        }

        public static int TargetFPS
        {
            get
            {
                return _targetFPS;
            }

            set
            {
                if (value >= 1 && value <= 2000)
                {
                    _targetFPS = value;
                }
            }
        }

        public static float TimeStep
        {
            get
            {
                return _timeStep;
            }

            set
            {
                if (value >= 0.0001f && value <= 1f)
                {
                    _timeStep = value;
                }
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

            set
            {
                _frameCount = value;
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


        public static float RecordTimeStep
        {
            get { return _recFrameTimeSpan; }
        }

        public static double RecordMaxSize
        {
            get { return _recSizeLimit; }
        }

        public static int ThreadsPerBlock
        {
            get
            {
                return _threadsPBExp;
            }

            set
            {
                if (value >= 4 && value <= Math.Log(MaxThreadsPerBlock, 2))
                {
                    _threadsPBExp = value;
                }
            }
        }

        public static int MaxThreadsPerBlock
        {
            get { return _maxThreadsPB; }

            set
            {
                if (value >= 4 && value <= 8192)
                {
                    _maxThreadsPB = value;
                }
            }
        }

        public static int RenderBurstFrames
        {
            get { return _burstSkips; }

            set
            {
                if (value >= 0 && value <= 20)
                {
                    _burstSkips = value;
                }
            }
        }

        public static float Viscosity
        {
            get { return _viscosity; }

            set
            {
                if (value >= 0 && value <= 1000)
                {
                    _viscosity = value;
                }
            }
        }

        public static double TotalTime
        {
            get { return _totalTime; }

            set
            {
                _totalTime = value;
            }
        }

        #endregion Public Properties

        private static float _viscosity = 10.0f;
        private const float _cullDistance = 20000; // Ultimately determines max grid index and mesh size, which ultimately determines a large portion of the GPU RAM usage. Increase with caution.
        private static int _cellSizeExp = 3;
        private static int _meshLevels = 4;
        private static int _threadsPBExp = 8;
        private static int _maxThreadsPB = DefaultThreadsPerBlock;
        private static int _burstSkips = 0;
        private static int _bursts = 0;
        private static int _targetFPS = 60;
        private static float _currentFPS = 0;
        private static float _peakFPS = 0;
        private static int _minFrameTime = 0;
        private static Int64 _frameCount = 0;
        private static double _totalTime = 0;
        private static float _timeStep = 0.008f;
        private static Average _avgFPS = new Average(40);
        private static ManualResetEventSlim _pausePhysicsWait = new ManualResetEventSlim(true);
        private static ManualResetEvent _stopLoopWait = new ManualResetEvent(true);
        private static ManualResetEventSlim _drawingDoneWait = new ManualResetEventSlim(true);

        private static bool _skipPhysics = false;
        private static bool _wasPaused = false;

        private static Task _loopTask;
        private static CancellationTokenSource _cancelTokenSource;
        private static Stopwatch _fpsTimer = new Stopwatch();

        private static float _recElapTime = 0f;
        private const float _recFrameTimeSpanDefault = 0.30f;
        private const double _recSizeLimitDefault = 0;
        private static float _recFrameTimeSpan = 0.30f;
        private static double _recSizeLimit = 0;

        private static Body[] _bodiesBuffer = new Body[0];

        private static IRecording _recorder = new IO.MessagePackRecorder();

        private static Stopwatch timer = new Stopwatch();

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
            _loopTask.Wait();
            _loopTask.Dispose();
            _drawingDoneWait.Wait(5000);
            PhysicsProvider.PhysicsCalc.Flush();
            _peakFPS = 0;
        }

        public static void End()
        {
            _cancelTokenSource.Cancel();
            _stopLoopWait.Reset();
            _stopLoopWait.WaitOne(2000);
            _loopTask.Wait();
            _loopTask.Dispose();
            PhysicsProvider.PhysicsCalc.Dispose();
            _peakFPS = 0;
        }

        /// <summary>
        /// Pause physics calculations and blocks calling thread until any current physics operation is complete.
        /// </summary>
        public static void WaitForPause()
        {
            _wasPaused = _skipPhysics;

            if (!_skipPhysics)
            {
                // Reset the wait handle.
                _pausePhysicsWait.Reset();
                // Wait until the handle is signaled after the GPU calcs complete.
                _pausePhysicsWait.Wait(2000);
            }
        }

        /// <summary>
        /// Resume physics calulations.
        /// </summary>
        public static void ResumePhysics(bool forceResume = false)
        {
            // If we were already paused at the last call, don't resume unless forced.
            // This is to keep other functions from overriding a manual pause and resuming unexpectedly.
            if (!_wasPaused || forceResume)
            {
                // Make sure the wait handle has been set
                // and set the skip bool to false to allow physics to be calculated again.

                if (!_recorder.PlaybackActive)
                {
                    _pausePhysicsWait.Set();
                    _skipPhysics = false;
                }
            }
        }

        private async static void DoLoop()
        {
            // Double-buffered body array:
            //
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
                        if (BodyManager.Bodies.Length > 1)
                        {
                            // Push the current field to rewind collection.
                            if (RewindBuffer)
                                BodyManager.PushState();

                            // 1.
                            // Copy the current bodies to another array.
                            _bodiesBuffer = new Body[BodyManager.Bodies.Length];
                            Array.Copy(BodyManager.Bodies, _bodiesBuffer, BodyManager.Bodies.Length);

                            // Calc all physics and movements.
                            PhysicsProvider.PhysicsCalc.CalcMovement(ref _bodiesBuffer, _timeStep, _viscosity, _cellSizeExp, _cullDistance, Collisions, _meshLevels, (int)Math.Pow(2, _threadsPBExp));

                            // 2.
                            // Wait for the drawing thread to complete if needed.
                            _drawingDoneWait.Wait(5000);

                            // 3.
                            // Copy the new data to the current body collection.
                            BodyManager.Bodies = _bodiesBuffer;
                           // BodyManager.Mesh = PhysicsProvider.PhysicsCalc.CurrentMesh;

                            // Do some final host-side processing.
                            BodyManager.PostProcess(RocheLimit);

                            // Increment physics frame count.
                            _frameCount++;
                            _totalTime += _timeStep;

                            // Send the data to the recorder if we are recording.
                            if (_recorder.RecordingActive && BodyManager.Bodies.Length > 0)
                            {
                                if (_recElapTime >= _recFrameTimeSpan)
                                {
                                    _recorder.RecordFrame(BodyManager.Bodies);
                                    _recElapTime = 0f;
                                }
                                _recElapTime += TimeStep;

                                if (_recSizeLimit > 0 && _recorder.FileSize >= _recSizeLimit)
                                {
                                    StopRecording();
                                    _skipPhysics = true;
                                }
                            }
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
                    _drawingDoneWait.Wait(5000);

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

                    FPSLimiter();

                    if (_bursts >= _burstSkips)
                    {
                        // 4.
                        // Draw the field asynchronously.
                        Renderer.DrawBodiesAsync(BodyManager.Bodies, DrawBodies, _drawingDoneWait);

                        _bursts = 0;
                    }

                    _bursts++;
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

        public static void StartRecording(string file, float timestep, double maxSize)
        {
            _recFrameTimeSpan = timestep;
            _recSizeLimit = maxSize * 1000000;

            _recorder.StopAll();

            _recorder.CreateRecording(file);
        }

        public static void StopRecording()
        {
            _recorder.StopAll();

            _recFrameTimeSpan = _recFrameTimeSpanDefault;
            _recSizeLimit = _recSizeLimitDefault;
        }

        public static double RecordedSize()
        {
            return _recorder.FileSize;
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

        private static void FPSLimiter()
        {
            long ticksPerSecond = TimeSpan.TicksPerSecond;
            float targetFrameTime = ticksPerSecond / (float)_targetFPS;
            long waitTime = 0;

            if (_fpsTimer.IsRunning)
            {
                long elapTime = _fpsTimer.Elapsed.Ticks;

                if (elapTime <= targetFrameTime)
                {
                    waitTime = (long)(targetFrameTime - elapTime);

                    if (waitTime > 0)
                    {
                        Thread.Sleep(new TimeSpan(waitTime));
                    }
                }

                float fps = ticksPerSecond / (elapTime + waitTime);

                _avgFPS.Add(fps);

                // Reset peak FPS every 100 frames.
                if (((_frameCount % 100) == 0))
                {
                    _peakFPS = fps;
                }

                _peakFPS = Math.Max(_peakFPS, fps);

                _currentFPS = (int)_avgFPS.Current;

                _fpsTimer.Restart();
            }
            else
            {
                _fpsTimer.Start();
                return;
            }
        }
    }
}