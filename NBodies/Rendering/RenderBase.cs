using NBodies.Extensions;
using NBodies.Helpers;
using NBodies.Physics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Linq;

namespace NBodies.Rendering
{
    public abstract class RenderBase
    {
        public static List<OverlayGraphic> OverLays = new List<OverlayGraphic>();
        public static bool AAEnabled = true;
        public static bool Trails = false;
        public static bool ClipView = true;
        public static bool ShowForce = false;
        public static bool ShowAllForce = false;
        public static bool ShowPath = false;
        public static bool ShowMesh = false;
        public static bool SortZOrder = true;
        public static bool FastPrimitives = true;
        public static DisplayStyle DisplayStyle = DisplayStyle.Normal;

        public static float StyleScaleMax
        {
            get
            {
                return _styleScales[(int)DisplayStyle];
            }

            set
            {
                if (value > 0 && value <= 8000)
                {
                    _styleScales[(int)DisplayStyle] = value;
                }
            }
        }

        public static int BodyAlpha
        {
            get
            {
                return _bodyAlpha;
            }

            set
            {
                if (value >= 1 && value <= 255)
                {
                    _bodyAlpha = value;
                }
            }
        }

        public Control TargetControl
        {
            get { return _targetControl; }
        }

        private static float _styleScaleMax = 210;
        private static int _bodyAlpha = 210;
        private static float[] _styleScales = new float[1] { _styleScaleMax };

        protected Control _targetControl;
        protected float _prevScale = 0;
        protected float _currentScale = 0;
        protected static Color _defaultClearColor = Color.Black;//Color.FromArgb(255, 23, 23, 25);//Color.Black;
        protected Color _clearColor = _defaultClearColor;
        protected RectangleF _cullTangle;
        protected Size _viewPortSize;

        private ManualResetEventSlim _orbitReadyWait = new ManualResetEventSlim(false);
        private bool _orbitOffloadRunning = false;
        private List<PointF> _orbitPath = new List<PointF>();
        private List<PointF> _drawPath = new List<PointF>();
        private bool _blurClearHack = false;

        private int[] _pointers = new int[0];
        private int[] _buckets = new int[0];

        private Stopwatch timer = new Stopwatch();

        protected RenderBase(Control targetControl)
        {
            Init(targetControl);
        }

        protected void Init(Control targetControl)
        {
            _targetControl = targetControl;
            InitGraphics();

            int styleCount = Enum.GetValues(typeof(DisplayStyle)).Cast<int>().Max() + 1;
            _styleScales = new float[styleCount];
            for (int i = 0; i < _styleScales.Length; i++)
                _styleScales[i] = _styleScaleMax;
        }

        public async Task DrawBodiesAsync(Body[] bodies, bool drawBodies, ManualResetEventSlim completeCallback)
        {
           // completeCallback.Reset();

            //await Task.Run(() =>
            //{
                BodyManager.RebuildUIDIndex();
                int maxUID = BodyManager.TopUID;
                bool overlayVisible = OverlaysVisible();
                var finalOffset = CalcFinalOffset();

                CheckScale();

                BeginDraw();

                if (!Trails || overlayVisible)
                {
                    Clear(_clearColor);
                    //     _blurClearHack = false;
                }

                if (drawBodies && bodies.Length > 0)
                {
                    // If trails are enabled, clear one frame with a slightly 
                    // off-black color to try to hide persistent artifacts
                    // left by the lame blur technique.
                    //if (Trails && !_blurClearHack)
                    //{
                    //    if (DisplayStyle != DisplayStyle.HighContrast)
                    //        Clear(Color.FromArgb(12, 12, 12));

                    //    _blurClearHack = true;
                    //}
                    //else if (!Trails && _blurClearHack)
                    //{
                    //    _blurClearHack = false;
                    //}

                    SetAntiAliasing(AAEnabled);

                    _cullTangle = new RectangleF(0 - finalOffset.X, 0 - finalOffset.Y, _viewPortSize.Width / ViewportOffsets.CurrentScale, _viewPortSize.Height / ViewportOffsets.CurrentScale);

                    // Since the bodies are being sorted by their spatial index
                    // we need to sort them (again) by a persistent value; we will use their UIDs.
                    // This is done because the spatial sorting can rapidly change the resulting
                    // z-order of the bodies, which causes flickering.

                    // Perform a Bucket Sort with the bodies UIDs.
                    // We will also cull the bodies outside the viewport.

                    int nVis = 0;

                    if (bodies.Length > 0)
                    {
                        if (SortZOrder)
                        {
                            int len = maxUID + 1;

                            //Realloc as needed.
                            if (_buckets.Length < len)
                                _buckets = new int[len];

                            if (_pointers.Length < bodies.Length)
                                _pointers = new int[bodies.Length];

                            // Clear buckets.
                            for (int i = 0; i < len; i++)
                            {
                                _buckets[i] = -1;
                            }

                            // Find bodies inside the viewport and put them in buckets by UID.
                            int maxVisUID = 0;
                            for (int i = 0; i < bodies.Length; i++)
                            {
                                if (_cullTangle.Contains(bodies[i].PosX, bodies[i].PosY))
                                {
                                    _buckets[bodies[i].UID] = i;
                                    maxVisUID = Math.Max(maxVisUID, bodies[i].UID); // Record max seen UID to save iterations of the following step.
                                }
                            }

                            // Populate pointers with the locations of populated buckets.
                            for (int i = 0; i <= maxVisUID; i++)
                            {
                                if (_buckets[i] != -1)
                                {
                                    _pointers[nVis] = i;
                                    nVis++;
                                }
                            }
                        }
                        else if (!SortZOrder && _buckets.Length > 0)
                        {
                            // Null the indexes if Z-sort is turned off.
                            _buckets = new int[0];
                            _pointers = new int[0];

                        }
                    }

                    int n = bodies.Length;

                    if (SortZOrder)
                        n = nVis;

                    //for (int i = 0; i < n; i++)
                    //{
                    //    Body body;

                    //    if (SortZOrder && _pointers.Length > 0)
                    //    {
                    //        body = bodies[_buckets[_pointers[i]]];
                    //    }
                    //    else
                    //    {
                    //        body = bodies[i];
                    //    }

                    //    var bodyLoc = new PointF((body.PosX + finalOffset.X), (body.PosY + finalOffset.Y));

                    //    if (ClipView && !SortZOrder)
                    //    {
                    //        if (!_cullTangle.Contains(body.PosX, body.PosY)) continue;
                    //    }

                    //    Color bodyColor = Color.White;

                    //    switch (DisplayStyle)
                    //    {
                    //        case DisplayStyle.Normal:
                    //            bodyColor = Color.FromArgb(BodyAlpha, Color.FromArgb(body.Color));
                    //            _clearColor = _defaultClearColor;

                    //            break;

                    //        case DisplayStyle.Pressure:
                    //            bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, StyleScaleMax, body.Pressure, true);
                    //            _clearColor = _defaultClearColor;

                    //            break;

                    //        case DisplayStyle.Density:
                    //            bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, StyleScaleMax, body.Density / body.Mass, true);
                    //            _clearColor = _defaultClearColor;

                    //            break;

                    //        case DisplayStyle.Velocity:
                    //            bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, StyleScaleMax, body.AggregateSpeed(), true);
                    //            _clearColor = _defaultClearColor;

                    //            break;

                    //        case DisplayStyle.Index:
                    //            bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, maxUID, body.UID, true);
                    //            _clearColor = _defaultClearColor;

                    //            break;

                    //        case DisplayStyle.SpatialOrder:
                    //            int orderIdx = 0;
                    //            if (SortZOrder)
                    //                orderIdx = _buckets[body.UID];
                    //            else
                    //                orderIdx = i;

                    //            bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, bodies.Length, orderIdx, true);

                    //            _clearColor = _defaultClearColor;

                    //            break;

                    //        case DisplayStyle.Force:
                    //            bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, StyleScaleMax, (body.ForceTot / body.Mass), true);
                    //            _clearColor = _defaultClearColor;

                    //            break;

                    //        case DisplayStyle.HighContrast:
                    //            bodyColor = Color.Black;
                    //            _clearColor = Color.White;

                    //            break;
                    //    }

                    //    //Draw body.
                    //    DrawBody(bodyColor, bodyLoc.X, bodyLoc.Y, body.Size, body.IsBlackHole);
                    //}

                    DrawBodiesRaw(bodies);

                    //if (Trails && !overlayVisible)
                    //    DrawBlur(Color.FromArgb(10, _clearColor));

                    if (ShowAllForce)
                    {
                        DrawForceVectors(bodies, finalOffset.X, finalOffset.Y);
                    }

                    if (BodyManager.FollowSelected)
                    {
                        var followBody = BodyManager.FollowBody();

                        if (ShowForce)
                        {
                            DrawForceVectors(new Body[] { followBody }, finalOffset.X, finalOffset.Y);
                        }

                        if (ShowPath)
                        {
                            // Start the offload task if needed.
                            if (!_orbitOffloadRunning)
                                CalcOrbitOffload();

                            // Previous orbit calc is complete.
                            // Bring the new data into another reference.
                            if (!_orbitReadyWait.Wait(0))
                            {
                                // Reference the new data.
                                _drawPath = _orbitPath;

                                _orbitReadyWait.Set();
                            }

                            // Add the final offset to the path points and draw them as a line.
                            if (_drawPath.Count > 0)
                            {
                                var pathArr = _drawPath.ToArray();

                                pathArr[0] = new PointF(followBody.PosX, followBody.PosY);

                                DrawOrbit(pathArr, finalOffset);
                            }
                        }
                    }
                }

                if (ShowMesh)
                {
                    DrawMesh(BodyManager.Mesh, finalOffset.X, finalOffset.Y);
                }

                if (overlayVisible)
                    DrawOverlays(finalOffset.X, finalOffset.Y);


                DrawStats(GetStats(), Color.FromArgb(255, 0, 192, 0), Color.FromArgb(100, _clearColor));

                EndDraw();
            //});

           // completeCallback.Set();
        }

        private string GetStats()
        {
            // Define a completely arbitrary yet slightly informative time span.
            var elapTime = TimeSpan.FromSeconds(MainLoop.TotalTime * 10000);

            string stats = $@"Renderer: {MainLoop.Renderer.ToString()}

FPS: {Math.Round(MainLoop.CurrentFPS, 2)} ({Math.Round(MainLoop.PeakFPS, 2)})
Count: {MainLoop.FrameCount}
Time: {elapTime.Days} days  {elapTime.Hours} hr  {elapTime.Minutes} min
Grid Passes: {OpenCLPhysics.GridPasses}

Bodies: {BodyManager.BodyCount}
Tot Mass: {Math.Round(BodyManager.TotalMass, 2)}

Scale: {Math.Round(ViewportOffsets.CurrentScale, 2)}";

            if (BodyManager.FollowSelected)
            {
                var body = BodyManager.FollowBody();

                stats += $@"

Density: {body.Density}
Press: {body.Pressure}
Agg. Speed: {body.AggregateSpeed()}";

            }

            if (MainLoop.Recorder.RecordingActive)
            {
                stats += $@"

Rec Size (MB): {Math.Round((MainLoop.RecordedSize() / (float)1000000), 2)}";

            }

            return stats;
        }


        public abstract void InitGraphics();

        public abstract void UpdateViewportSize(float width, float height);

        public abstract void UpdateGraphicsScale(float currentScale);

        public abstract void Clear(Color color);

        public abstract void SetAntiAliasing(bool enabled);

        public abstract void DrawBody(Color color, float X, float Y, float size, bool isBlackHole);

        public abstract void DrawBodiesRaw(Body[] bodies);


        public abstract void DrawForceVectors(Body[] bodies, float offsetX, float offsetY);

        public abstract void DrawMesh(MeshCell[] mesh, float offsetX, float offsetY);

        public abstract void DrawOverlays(float offsetX, float offsetY);

        public abstract void DrawOrbit(PointF[] points, PointF finalOffset);

        public abstract void DrawBlur(Color color);

        public abstract void DrawStats(string stats, System.Drawing.Color foreColor, System.Drawing.Color backColor);

        public abstract void BeginDraw();

        public abstract void EndDraw();

        public abstract void Destroy();

        protected internal bool OverlaysVisible()
        {
            if (OverLays.Count < 1)
                return false;

            for (int i = 0; i < OverLays.Count; i++)
            {
                if (OverLays[i].Visible)
                    return true;
            }

            return false;
        }

        protected internal void CheckScale()
        {
            _currentScale = ViewportOffsets.CurrentScale;

            if (_targetControl.ClientSize != _viewPortSize)
            {
                UpdateViewportSize(_targetControl.ClientSize.Width, _targetControl.ClientSize.Height);
                UpdateGraphicsScale(_currentScale);
                _blurClearHack = false;
                _viewPortSize = _targetControl.ClientSize;
            }

            if (_prevScale != _currentScale)
            {
                UpdateGraphicsScale(_currentScale);

                _prevScale = _currentScale;
            }
        }

        protected internal PointF CalcFinalOffset()
        {
            if (BodyManager.FollowSelected)
            {
                Body followBody = BodyManager.FollowBody();
                ViewportOffsets.ViewportOffset.X = -followBody.PosX;
                ViewportOffsets.ViewportOffset.Y = -followBody.PosY;
            }

            var finalOffset = ViewportOffsets.ViewportOffset.Add(ViewportOffsets.ScaleOffset);

            return finalOffset;
        }

        // Offload orbit calcs to another task/thread and update the renderer periodically.
        // We don't want to block the rendering task while we wait for this calc to finish.
        // We just populate a local variable with the new data and signal the render thread
        // to update its data for drawing.
        private async Task CalcOrbitOffload()
        {
            if (_orbitOffloadRunning)
                return;

            _orbitReadyWait.Set();

            await Task.Run(() =>
            {
                _orbitOffloadRunning = true;

                while (BodyManager.FollowSelected)
                {
                    _orbitReadyWait.Wait(-1);

                    _orbitPath = BodyManager.CalcPathCircle(BodyManager.FollowBody());

                    //_orbitPath = BodyManager.CalcPath(BodyManager.FollowBody());
                    // _orbitPath = BodyManager.CalcPathCM(BodyManager.FollowBody());

                    _orbitReadyWait.Reset();
                }

                _orbitOffloadRunning = false;
            });
        }

        internal Color GetVariableColor(Color startColor, Color endColor, float maxValue, float currentValue, bool translucent = false)
        {
            const int maxIntensity = 255;
            float intensity = 0;
            long r1, g1, b1, r2, g2, b2;

            r1 = startColor.R;
            g1 = startColor.G;
            b1 = startColor.B;

            r2 = endColor.R;
            g2 = endColor.G;
            b2 = endColor.B;

            if (currentValue > 0)
            {
                // Compute the intensity of the end color.
                intensity = (maxIntensity / (maxValue / currentValue));
            }

            // Clamp the intensity within the max.
            if (intensity > maxIntensity) intensity = maxIntensity;

            // Calculate the new RGB values from the intensity.
            int newR, newG, newB;
            newR = (int)(r1 + (r2 - r1) / (float)maxIntensity * intensity);
            newG = (int)(g1 + (g2 - g1) / (float)maxIntensity * intensity);
            newB = (int)(b1 + (b2 - b1) / (float)maxIntensity * intensity);

            if (translucent)
            {
                return Color.FromArgb(BodyAlpha, newR, newG, newB);
            }
            else
            {
                return Color.FromArgb(newR, newG, newB);
            }
        }

        internal Color GetVariableColor(Color startColor, Color midColor, Color endColor, float maxValue, float currentValue, bool translucent = false)
        {
            const int maxIntensity = 255;
            float intensity = 0;
            long r1 = 0;
            long g1 = 0;
            long b1 = 0;
            long r2 = 0;
            long g2 = 0;
            long b2 = 0;

            if (currentValue <= (maxValue / 2f))
            {
                r1 = startColor.R;
                g1 = startColor.G;
                b1 = startColor.B;

                r2 = midColor.R;
                g2 = midColor.G;
                b2 = midColor.B;

                maxValue = maxValue / 2f;
            }
            else
            {
                r1 = midColor.R;
                g1 = midColor.G;
                b1 = midColor.B;

                r2 = endColor.R;
                g2 = endColor.G;
                b2 = endColor.B;

                maxValue = maxValue / 2f;
                currentValue = currentValue - maxValue;
            }

            if (currentValue > 0)
            {
                // Compute the intensity of the end color.
                intensity = (maxIntensity / (maxValue / currentValue));
            }

            // Clamp the intensity within the max.
            if (intensity > maxIntensity) intensity = maxIntensity;

            // Calculate the new RGB values from the intensity.
            int newR, newG, newB;
            newR = (int)(r1 + (r2 - r1) / (float)maxIntensity * intensity);
            newG = (int)(g1 + (g2 - g1) / (float)maxIntensity * intensity);
            newB = (int)(b1 + (b2 - b1) / (float)maxIntensity * intensity);

            if (translucent)
            {
                return Color.FromArgb(BodyAlpha, newR, newG, newB);
            }
            else
            {
                return Color.FromArgb(newR, newG, newB);
            }
        }

        public static void AddOverlay(OverlayGraphic overlay)
        {
            if (!OverLays.Contains(overlay))
            {
                OverLays.Add(overlay);
            }
        }
    }
}