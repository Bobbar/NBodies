using NBodies.Physics;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NBodies.Rendering
{
    public static class Renderer
    {
        public static List<OverlayGraphic> OverLays = new List<OverlayGraphic>();
        public static bool AntiAliasing = true;
        public static bool Trails = false;
        public static bool ClipView = true;
        public static bool ShowForce = false;
        public static bool ShowPath = false;
        public static float PressureScaleMax = 150;

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

        public static DisplayStyle DisplayStyle = DisplayStyle.Normal;

        private static ManualResetEventSlim _orbitReadyWait = new ManualResetEventSlim(false);
        private static List<PointF> _orbitPath = new List<PointF>();
        private static List<PointF> _drawPath = new List<PointF>();
        private static bool _orbitOffloadRunning = false;

        private static int _bodyAlpha = 210;
        private static BufferedGraphicsContext _currentContext;
        private static BufferedGraphics _buffer;
        private static PictureBox _imageControl;
        private static float _prevScale = 0;
        private static Pen _blackHoleStroke = new Pen(Color.Red);
        private static Pen _forcePen = new Pen(Color.White, 0.2f);
        private static Pen _orbitPen = new Pen(Color.FromArgb(200, Color.White), 0.4f) { EndCap = LineCap.ArrowAnchor };//new Pen(Color.White, 0.4f) { DashStyle = DashStyle.Dot, EndCap = LineCap.ArrowAnchor };

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

            //_buffer.Graphics.CompositingMode = CompositingMode.SourceOver;
            //_buffer.Graphics.CompositingQuality = CompositingQuality.HighSpeed;
            //_buffer.Graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
            //_buffer.Graphics.PixelOffsetMode = PixelOffsetMode.HighSpeed;

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

        private static System.Diagnostics.Stopwatch timer = new System.Diagnostics.Stopwatch();

        public async static Task DrawBodiesAsync(Body[] bodies, ManualResetEventSlim completeCallback)
        {
            completeCallback.Reset();

            await Task.Run(() =>
             {
                 Body followBody = new Body();

                 if (BodyManager.FollowSelected)
                 {
                     followBody = BodyManager.FollowBody();
                     var followOffset = new PointF(followBody.LocX, followBody.LocY);
                     RenderVars.ViewportOffset.X = -followOffset.X;
                     RenderVars.ViewportOffset.Y = -followOffset.Y;
                 }

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

                 _cullTangle = new RectangleF(0 - finalOffset.X, 0 - finalOffset.Y, _imageControl.Size.Width / RenderVars.CurrentScale, _imageControl.Size.Height / RenderVars.CurrentScale);

                 for (int i = 0; i < bodies.Length; i++)
                 {
                     var body = bodies[i];

                     if (body.Visible == 1)
                     {
                         if (ClipView)
                         {
                             if (!_cullTangle.Contains(body.LocX, body.LocY)) continue;
                         }

                         Color bodyColor = new Color();

                         switch (DisplayStyle)
                         {
                             case DisplayStyle.Normal:
                                 bodyColor = Color.FromArgb(_bodyAlpha, Color.FromArgb(body.Color));
                                 _spaceColor = Color.Black;
                                 break;

                             case DisplayStyle.Pressures:
                                 bodyColor = GetVariableColor(Color.Blue, Color.Red, PressureScaleMax, body.Pressure, true);
                                 _spaceColor = Color.Black;
                                 break;

                             case DisplayStyle.Speeds:
                                 bodyColor = GetVariableColor(Color.Blue, Color.Red, PressureScaleMax, body.AggregateSpeed(), true);
                                 _spaceColor = Color.Black;
                                 break;

                             case DisplayStyle.Forces:
                                 bodyColor = GetVariableColor(Color.Blue, Color.Red, PressureScaleMax, (body.ForceTot / body.Mass), true);
                                 _spaceColor = Color.Black;
                                 break;

                             case DisplayStyle.HighContrast:
                                 bodyColor = Color.Black;
                                 _spaceColor = Color.White;
                                 break;
                         }

                         if (BodyManager.FollowSelected)
                         {
                             if (BodyManager.FollowBody().UID == body.UID)
                             {
                                 followBody = body;
                                 bodyColor = Color.Red;
                             }
                         }
                         else
                         {
                             followBody = new Body();
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

                     if (ShowForce)
                     {
                         var f = new PointF(followBody.ForceX, followBody.ForceY);
                         //  var f = new PointF(followBody.SpeedX, followBody.SpeedY);
                         var bloc = new PointF(followBody.LocX, followBody.LocY);
                         f = f.Multi(0.1f);
                         var floc = bloc.Add(f);
                         _buffer.Graphics.DrawLine(_forcePen, bloc.Add(finalOffset), floc.Add(finalOffset));
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

                             pathArr[0] = new PointF(followBody.LocX, followBody.LocY);


                             DrawOrbit(pathArr, finalOffset);
                             //for (int a = 0; a < pathArr.Length; a++)
                             //{
                             //    pathArr[a] = pathArr[a].Add(finalOffset);
                             //}

                             //_buffer.Graphics.DrawLines(_orbitPen, pathArr);
                         }
                     }
                 }

                 DrawOverlays(finalOffset);


                 if (!_imageControl.IsDisposed && !_imageControl.Disposing)
                     _buffer.Render();
             });

            completeCallback.Set();
        }

        private static void DrawOrbit(PointF[] points, PointF finalOffset)
        {
            if (points.Length < 1)
                return;

            for (int a = 0; a < points.Length; a++)
            {
                points[a] = points[a].Add(finalOffset);
            }

            _buffer.Graphics.DrawLines(_orbitPen, points);
        }

        // Offload orbit calcs to another task/thread and update the renderer periodically.
        // We don't want to block the rendering task while we wait for this calc to finish.
        // We just populate a local variable with the new data and signal the render thread
        // to update its data for drawing.
        private async static Task CalcOrbitOffload()
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

                    _orbitPath = BodyManager.CalcPath(BodyManager.FollowBody());
                    // _orbitPath = BodyManager.CalcPathCM(BodyManager.FollowBody());

                    _orbitReadyWait.Reset();
                }

                _orbitOffloadRunning = false;
            });
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
                return Color.FromArgb(_bodyAlpha, newR, newG, newB);
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

        public static void AddOverlay(OverlayGraphic overlay)
        {
            if (!OverLays.Contains(overlay))
            {
                OverLays.Add(overlay);
            }
        }

        private static void DrawOverlays(PointF finalOffset)
        {


            foreach (var overlay in OverLays.ToArray())
            {
                if (overlay.Visible)
                {
                    switch (overlay.Type)
                    {
                        case OverlayGraphicType.Orbit:
                            DrawOrbit(overlay.OrbitPath.ToArray(), finalOffset);
                            break;
                    }
                }
            }



            var ogSt = _buffer.Graphics.Save();
            _buffer.Graphics.ResetTransform();
            foreach (var overlay in OverLays.ToArray())
            {
                if (overlay.Visible)
                {
                    switch (overlay.Type)
                    {
                        case OverlayGraphicType.Text:
                            _buffer.Graphics.DrawString(overlay.Value, _infoTextFont, Brushes.White, overlay.Location);
                            break;

                        case OverlayGraphicType.Line:
                            _buffer.Graphics.DrawLine(new Pen(Color.White) { EndCap = LineCap.ArrowAnchor }, overlay.Location, overlay.Location2);
                            break;
                    }
                }

            }

            _buffer.Graphics.Restore(ogSt);
        }

        //private static void DrawInfoText(PointF bodyLoc, CUDAMain.Body body)
        //{
        //    var textLoc = new PointF((float)(bodyLoc.X + (body.Size * 0.5f)), (float)(bodyLoc.Y - (body.Size * 0.5f)));
        //    _buffer.Graphics.ResetTransform();
        //    _buffer.Graphics.DrawString(string.Format("SpeedX: {0}", body.SpeedX), infoTextFont, Brushes.White, textLoc);
        //}
    }
}