using NBodies.Physics;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using System.Diagnostics;
using System;
using System.Linq;

namespace NBodies.Rendering
{
    public static class Renderer
    {
        public static List<OverlayGraphic> OverLays = new List<OverlayGraphic>();
        public static bool AntiAliasing = true;
        public static bool Trails = false;
        public static bool ClipView = true;
        public static bool ShowForce = false;
        public static float PressureScaleMax = 150;

        public static DisplayStyle DisplayStyle = DisplayStyle.Normal;

        private static BufferedGraphicsContext _currentContext;
        private static BufferedGraphics _buffer;

        private static PictureBox _imageControl;
        private static float _prevScale = 0;

        private static Pen _blackHoleStroke = new Pen(Color.Red);
        private static Pen _forcePen = new Pen(Color.White, 0.2f);
        private static Color _spaceColor = Color.Black;

        private static Size _prevSize = new Size();

        private static Font _infoTextFont = new Font("Tahoma", 8, FontStyle.Regular);

        private static RectangleF _cullTangle;

        public static void Init(PictureBox imageControl)
        {
            _imageControl = imageControl;
            _forcePen.EndCap = LineCap.ArrowAnchor;
        }

        private static void InitGfx()
        {
            _currentContext = BufferedGraphicsManager.Current;
            _buffer = _currentContext.Allocate(_imageControl.CreateGraphics(), _imageControl.DisplayRectangle);
            _prevSize = _imageControl.Size;
        }

        public static void CheckScale()
        {
            if (_imageControl.Size != _prevSize)
            {
                InitGfx();

                UpdateGraphicsScale();
            }

            if (_prevScale != RenderVars.CurrentScale)
            {
                UpdateGraphicsScale();
            }
        }

        private static void UpdateGraphicsScale()
        {
            _buffer.Graphics.ResetTransform();
            _buffer.Graphics.ScaleTransform(RenderVars.CurrentScale, RenderVars.CurrentScale);
            _prevScale = RenderVars.CurrentScale;
        }

        public static void DrawBodies(Body[] bodies)
        {
            var finalOffset = PointHelper.Add(RenderVars.ViewportOffset, RenderVars.ScaleOffset);
            CheckScale();

            if (!Trails) _buffer.Graphics.Clear(_spaceColor);

            if (AntiAliasing)
            {
                _buffer.Graphics.SmoothingMode = SmoothingMode.HighQuality;
            }
            else
            {
                _buffer.Graphics.SmoothingMode = SmoothingMode.None;
            }

           
            for (int i = 0; i < bodies.Length; i++)
            {

                var body = bodies[i];

                if (body.Visible == 1)
                {
                    if (ClipView)
                    {
                        _cullTangle = new RectangleF(0 - finalOffset.X, 0 - finalOffset.Y, _imageControl.Size.Width / RenderVars.CurrentScale, _imageControl.Size.Height / RenderVars.CurrentScale);
                        if (!_cullTangle.Contains(body.LocX, body.LocY)) continue;
                    }

                    Color bodyColor = new Color();

                    switch (DisplayStyle)
                    {
                        case DisplayStyle.Normal:
                            bodyColor = Color.FromArgb(body.Color);
                            _spaceColor = Color.Black;
                            break;

                        case DisplayStyle.Pressures:
                            //bodyColor = GetVariableColor(Color.Blue, Color.Red, 500, (int)body.Density);
                           // bodyColor = GetVariableColor(Color.Blue, Color.Red, maxPressure, body.Pressure, true);
                           bodyColor = GetVariableColor(Color.Blue, Color.Red, PressureScaleMax, body.Pressure, true);

                            _spaceColor = Color.Black;
                            break;

                        case DisplayStyle.HighContrast:
                            bodyColor = Color.Black;
                            _spaceColor = Color.White;
                            break;
                    }

                    using (var bodyBrush = new SolidBrush(bodyColor))
                    {
                        var bodyLoc = new PointF((body.LocX - body.Size * 0.5f + finalOffset.X), (body.LocY - body.Size * 0.5f + finalOffset.Y));

                        //Draw body.
                        _buffer.Graphics.FillEllipse(bodyBrush, bodyLoc.X, bodyLoc.Y, body.Size, body.Size);

                        // If blackhole, stroke with red circle.
                        if (body.BlackHole == 1)
                        {
                            _buffer.Graphics.DrawEllipse(_blackHoleStroke, bodyLoc.X, bodyLoc.Y, body.Size, body.Size);
                        }
                    }
                }
            }

            if (BodyManager.FollowSelected)
            {
                var body = BodyManager.FollowBody();
                var followOffset = BodyManager.FollowBodyLoc();
                RenderVars.ViewportOffset.X = -followOffset.X;
                RenderVars.ViewportOffset.Y = -followOffset.Y;


                if (ShowForce)
                {
                    var f = new PointF(body.ForceX, body.ForceY);
                    //  var f = new PointF(body.SpeedX, body.SpeedY);
                    var bloc = new PointF(body.LocX, body.LocY);
                    f = f.Multi(0.1f);
                    var floc = bloc.Add(f);
                    _buffer.Graphics.DrawLine(_forcePen, bloc.Add(finalOffset), floc.Add(finalOffset));
                }

            }


            //    DrawOverlays();
            if (!_imageControl.IsDisposed && !_imageControl.Disposing)
                _buffer.Render();
        }

        private static Color GetVariableColor(Color startColor, Color endColor, float maxValue, float currentValue, bool translucent = false)
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
                return Color.FromArgb(200, newR, newG, newB);
            }
            else
            {
                return Color.FromArgb(newR, newG, newB);
            }
        }

        private static float GetMaxPressure(Body[] bodies)
        {
            var blist = bodies.ToList();
            float maxPress = 0.0f;

            blist.ForEach(b =>
            {
                if (b.Pressure > maxPress)
                    maxPress = b.Pressure;
            });

            return maxPress;
        }


        private static void DrawOverlays()
        {
            var ogSt = _buffer.Graphics.Save();

            foreach (var overlay in OverLays)
            {
                switch (overlay.Type)
                {
                    case OverlayGraphicType.Text:
                        _buffer.Graphics.ResetTransform();
                        _buffer.Graphics.DrawString(overlay.Value, _infoTextFont, Brushes.White, overlay.Location);
                        break;
                }
            }

            _buffer.Graphics.Restore(ogSt);
            OverLays.Clear();
        }

        //private static void DrawInfoText(PointF bodyLoc, CUDAMain.Body body)
        //{
        //    var textLoc = new PointF((float)(bodyLoc.X + (body.Size * 0.5f)), (float)(bodyLoc.Y - (body.Size * 0.5f)));
        //    _buffer.Graphics.ResetTransform();
        //    _buffer.Graphics.DrawString(string.Format("SpeedX: {0}", body.SpeedX), infoTextFont, Brushes.White, textLoc);
        //}
    }
}