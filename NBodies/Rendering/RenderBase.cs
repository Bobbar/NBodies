﻿using NBodies.Extensions;
using NBodies.Helpers;
using NBodies.Physics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

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
                return _styleScaleMax;
            }

            set
            {
                if (value > 0 && value <= 2000)
                {
                    _styleScaleMax = value;
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

        private static float _styleScaleMax = 210;
        private static int _bodyAlpha = 210;

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

        private Stopwatch timer = new Stopwatch();

        protected RenderBase(Control targetControl)
        {
            Init(targetControl);
        }

        protected void Init(Control targetControl)
        {
            _targetControl = targetControl;
            InitGraphics();
        }

        public async Task DrawBodiesAsync(Body[] bodies, ManualResetEventSlim completeCallback)
        {
            completeCallback.Reset();

            await Task.Run(() =>
            {
                bool overlayVisible = OverlaysVisible();
                var finalOffset = CalcFinalOffset();

                CheckScale();

                BeginDraw();

                if (!Trails || overlayVisible)
                {
                    Clear(_clearColor);
                    _blurClearHack = false;
                }

                // If trails are enabled, clear one frame with a slightly 
                // off-black color to try to hide persistent artifacts
                // left by the lame blur technique.
                if (Trails && !_blurClearHack)
                {
                    Clear(Color.FromArgb(12, 12, 12));
                    _blurClearHack = true;
                }
                else if (!Trails && _blurClearHack)
                {
                    _blurClearHack = false;
                }

                SetAntiAliasing(AAEnabled);

                _cullTangle = new RectangleF(0 - finalOffset.X, 0 - finalOffset.Y, _viewPortSize.Width / ViewportOffsets.CurrentScale, _viewPortSize.Height / ViewportOffsets.CurrentScale);

                // Since the bodies are being sorted by their spatial index
                // we need to sort them (again) by a persistent value; we will use their UIDs.
                // This is done because the spatial sorting can rapidly change the resulting
                // z-order of the bodies, which causes flickering.

                // Collect the index ID, and UIDs into two arrays,
                // then sort them together to provide an ordered "lookup" array.

                int[] bodyIds = new int[0];
                int[] bodyUids = new int[0];

                if (SortZOrder)
                {
                    bodyIds = new int[bodies.Length];
                    bodyUids = new int[bodies.Length];

                    for (int i = 0; i < bodies.Length; ++i)
                    {
                        bodyIds[i] = i;
                        bodyUids[i] = bodies[i].UID;
                    }

                    Array.Sort(bodyUids, bodyIds);
                }

                for (int i = 0; i < bodies.Length; i++)
                {
                    Body body;

                    if (SortZOrder && bodyIds.Length > 0)
                    {
                        body = bodies[bodyIds[i]];
                    }
                    else
                    {
                        body = bodies[i];
                    }

                    var bodyLoc = new PointF((body.PosX + finalOffset.X), (body.PosY + finalOffset.Y));

                    if (body.Visible == 1)
                    {
                        if (ClipView)
                        {
                            if (!_cullTangle.Contains(body.PosX, body.PosY)) continue;
                        }

                        Color bodyColor = Color.White;

                        switch (DisplayStyle)
                        {
                            case DisplayStyle.Normal:
                                bodyColor = Color.FromArgb(BodyAlpha, Color.FromArgb(body.Color));
                                _clearColor = _defaultClearColor;

                                break;

                            case DisplayStyle.Pressures:
                                bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, StyleScaleMax, body.Pressure, true);
                                _clearColor = _defaultClearColor;

                                break;

                            case DisplayStyle.Speeds:
                                bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, StyleScaleMax, body.AggregateSpeed(), true);
                                _clearColor = _defaultClearColor;

                                break;

                            case DisplayStyle.Index:
                                bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, bodies.Length, i, true);
                                _clearColor = _defaultClearColor;

                                break;

                            case DisplayStyle.Forces:
                                bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, StyleScaleMax, (body.ForceTot / body.Mass), true);
                                _clearColor = _defaultClearColor;

                                break;

                            case DisplayStyle.HighContrast:
                                bodyColor = Color.Black;
                                _clearColor = Color.White;

                                break;
                        }

                        //Draw body.
                        DrawBody(body, bodyColor, bodyLoc.X, bodyLoc.Y, body.Size);
                    }
                }

                if (Trails && !overlayVisible)
                    DrawBlur(Color.FromArgb(10, _clearColor));

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

                if (ShowMesh)
                {
                    DrawMesh(BodyManager.Mesh, finalOffset.X, finalOffset.Y);
                }

                if (overlayVisible)
                    DrawOverlays(finalOffset.X, finalOffset.Y);

                EndDraw();
            });

            completeCallback.Set();
        }

        public abstract void InitGraphics();

        public abstract void UpdateViewportSize(float width, float height);

        public abstract void UpdateGraphicsScale(float currentScale);

        public abstract void Clear(Color color);

        public abstract void SetAntiAliasing(bool enabled);

        public abstract void DrawBody(Body body, Color color, float X, float Y, float size);

        public abstract void DrawForceVectors(Body[] bodies, float offsetX, float offsetY);

        public abstract void DrawMesh(MeshCell[] mesh, float offsetX, float offsetY);

        public abstract void DrawOverlays(float offsetX, float offsetY);

        public abstract void DrawOrbit(PointF[] points, PointF finalOffset);

        public abstract void DrawBlur(Color color);

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