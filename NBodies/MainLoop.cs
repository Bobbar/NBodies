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
using OpenTK;

namespace NBodies
{
    public static class MainLoop
    {
        public static bool DrawBodies = true;
        public static bool RocheLimit = true;
        public static bool Collisions = true;
        public static bool SyncRenderer = false;
        public const int DefaultThreadsPerBlock = 256;
        public static RenderBase Renderer;

        public static GLControl GLRenderer;

        #region Public Properties
        public static bool RewindBuffer
        {
            get { return _rewindBuffer; }

            set
            {
                if (_rewindBuffer != value)
                {
                    _rewindBuffer = value;

                    if (!_rewindBuffer)
                        BodyManager.ClearRewinder();
                }
            }
        }

        public static float KernelSize
        {
            get { return _kernelSize; }

            set
            {
                if (value > 0.1f && value <= 2.0f)
                {
                    _kernelSize = value;
                }
            }
        }

        public static float GasK
        {
            get { return _gasK; }

            set
            {
                // TODO: Clamp this to some range.
                _gasK = value;
            }
        }

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


        private static float _timeStep = 0.005f;
        private static float _kernelSize = 1.0f;
        private static float _viscosity = 15.0f;
        private static float _gasK = 0.3f;
        private const float _cullDistance = 30000; // Ultimately determines max grid index and mesh size, which ultimately determines a large portion of the GPU RAM usage. Increase with caution.
        private static int _cellSizeExp = 3;
        private static int _meshLevels = 4;
        private static int _threadsPBExp = 8;
        private static int _maxThreadsPB = DefaultThreadsPerBlock;
        private static int _targetFPS = 60;
        private static int _pauseFPS = 60;
        private static float _currentFPS = 0;
        private static float _peakFPS = 0;
        private static int _minFrameTime = 0;
        private static Int64 _frameCount = 0;
        private static double _totalTime = 0;
        private static int _skippedFrames = 0;

        private static Average _avgFPS = new Average(40);
        private static ManualResetEventSlim _pausePhysicsWait = new ManualResetEventSlim(true);
        private static ManualResetEventSlim _stopLoopWait = new ManualResetEventSlim(true);
        private static ManualResetEventSlim _renderReadyWait = new ManualResetEventSlim(true);

        private static bool _skipPhysics = false;
        private static bool _wasPaused = false;
        private static bool _rewindBuffer = false;

        private static Task _loopTask;
        private static CancellationTokenSource _cancelTokenSource;
        private static Stopwatch _fpsTimer = new Stopwatch();

        private static float _recElapTime = 0f;
        private const float _recFrameTimeSpanDefault = 0.30f;
        private const double _recSizeLimitDefault = 0;
        private static float _recFrameTimeSpan = 0.30f;
        private static double _recSizeLimit = 0;

        private static Body[] _bodiesBuffer = new Body[0];
        private static SimSettings _settings = new SimSettings();

        private static IRecording _recorder = new IO.MessagePackRecorder();

        private static Stopwatch timer = new Stopwatch();

        public static void StartLoop()
        {
            _minFrameTime = 1000 / TargetFPS;

            _cancelTokenSource = new CancellationTokenSource();
            _stopLoopWait.Set();

            SyncPhysicsBuffer();

            _loopTask = new Task(DoLoop, _cancelTokenSource.Token, TaskCreationOptions.LongRunning);
            _loopTask.Start();
        }

        public static void Stop()
        {
            _cancelTokenSource.Cancel();
            _stopLoopWait.Reset();
            _stopLoopWait.Wait(2000);
            _loopTask.Wait();
            _loopTask.Dispose();
            _renderReadyWait.Wait(5000);
            PhysicsProvider.PhysicsCalc.Flush();
            _peakFPS = 0;
        }

        public static void End()
        {
            _cancelTokenSource?.Cancel();
            _stopLoopWait?.Reset();
            _stopLoopWait?.Wait(2000);
            _loopTask?.Wait();
            _loopTask?.Dispose();
            PhysicsProvider.PhysicsCalc?.Dispose();
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

                // Rebuild UID index to ensure correct mouse to body UI behaviour.
                BodyManager.RebuildUIDIndex();
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
                    SyncPhysicsBuffer();

                    _pausePhysicsWait.Set();
                    _skipPhysics = false;
                }
            }
        }

        private static void DoLoop()
        {
            try
            {
                while (!_cancelTokenSource.IsCancellationRequested)
                {
                    if (!_skipPhysics && !_recorder.PlaybackActive)
                    {
                        if (_bodiesBuffer.Length > 1)
                        {
                            // Push the current field to rewind collection.
                            if (RewindBuffer)
                                BodyManager.PushState(_bodiesBuffer);

                            // True if post processing is needed.
                            // GPU kernels set the flag if any bodies need removed/fractured.
                            bool postNeeded = false;

                            // Calc all physics and movements.
                            PhysicsProvider.PhysicsCalc.CalcMovement(ref _bodiesBuffer, GetSettings(), (int)Math.Pow(2, _threadsPBExp), out postNeeded);

                            // Do some final host-side processing. (Remove culled, roche fractures, etc)
                            BodyManager.PostProcessFrame(ref _bodiesBuffer, RocheLimit, postNeeded);

                            // Increment physics frame count.
                            _frameCount++;
                            _totalTime += _timeStep;

                            // Send the data to the recorder if we are recording.
                            if (_recorder.RecordingActive && _bodiesBuffer.Length > 0)
                            {
                                if (_recElapTime >= _recFrameTimeSpan)
                                {
                                    _recorder.RecordFrame(_bodiesBuffer);
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
                    if (SyncRenderer)
                        _renderReadyWait.Wait(5000);

                    // If we are playing back a recording, get the current field frame
                    // from the recorder and bring it in to be rendered.
                    if (_recorder.PlaybackActive && !_recorder.PlaybackComplete)
                    {
                        var frame = _recorder.GetNextFrame();

                        if (frame != null && frame.Length > 0)
                        {
                            BodyManager.Bodies = frame;
                        }
                    }

                    if (DrawBodies)
                    {
                        // Check if renderer is ready for a new frame.
                        //if (_renderReadyWait.IsSet)
                        //{
                        //    _renderReadyWait.Reset();

                        // Get the most recent frame from the physics buffer.
                        if (!_skipPhysics)
                        {
                            if (BodyManager.Bodies.Length != _bodiesBuffer.Length)
                                BodyManager.Bodies = new Body[_bodiesBuffer.Length];
                            Array.Copy(_bodiesBuffer, 0, BodyManager.Bodies, 0, _bodiesBuffer.Length);
                        }

                        // Draw the field asynchronously.
                        //if (Renderer.TargetControl != null && Renderer.TargetControl.InvokeRequired)
                        //{
                        //    var del = new Action(() => Renderer.DrawBodiesAsync(BodyManager.Bodies, DrawBodies, _renderReadyWait));
                        //    Renderer.TargetControl.BeginInvoke(del);

                        //}


                        if (GLRenderer != null && GLRenderer.InvokeRequired)
                        {
                            var del = new Action(() => GLRenderer.Invalidate());
                            GLRenderer.BeginInvoke(del);

                        }

                        //  Renderer.DrawBodiesAsync(BodyManager.Bodies, DrawBodies, _renderReadyWait);
                        //    _skippedFrames = 0;
                        //}
                        //else
                        //{
                        //    _skippedFrames++;
                        //}
                    }

                    // Fixed FPS limit while paused.
                    if (!_skipPhysics && _bodiesBuffer.Length > 1)
                        FPSLimiter(_targetFPS);
                    else
                        FPSLimiter(_pauseFPS);

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

        private static void SyncPhysicsBuffer()
        {
            if (_bodiesBuffer.Length != BodyManager.Bodies.Length)
                _bodiesBuffer = new Body[BodyManager.Bodies.Length];

            Array.Copy(BodyManager.Bodies, 0, _bodiesBuffer, 0, BodyManager.Bodies.Length);
        }

        public static SimSettings GetSettings()
        {
            _settings.KernelSize = _kernelSize;
            _settings.DeltaTime = _timeStep;
            _settings.Viscosity = _viscosity;
            _settings.GasK = _gasK;
            _settings.CullDistance = _cullDistance;
            _settings.CollisionsOn = Convert.ToInt32(Collisions);
            _settings.MeshLevels = _meshLevels;
            _settings.CellSizeExponent = _cellSizeExp;
            return _settings;
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

            BodyManager.ReplaceBodies(_recorder.GetNextFrame());
            _recorder.SeekIndex = 0;

            StartLoop();

            return _recorder;
        }

        public static void StopPlayback()
        {
            _recorder.StopAll();
        }

        private static void FPSLimiter(int targetFPS)
        {
            long ticksPerSecond = TimeSpan.TicksPerSecond;
            float targetFrameTime = ticksPerSecond / (float)targetFPS;
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