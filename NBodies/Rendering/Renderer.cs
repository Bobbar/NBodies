using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using System.Collections.Generic;
using NBodies.Physics;

namespace NBodies.Rendering
{
    public static class Renderer
    {
        public static List<OverlayGraphic> OverLays = new List<OverlayGraphic>();
        public static bool AntiAliasing = true;
        public static bool Trails = false;

        public static bool HighContrast
        {
            get
            {
                return _highContrast;
            }

            set
            {
                _highContrast = value;

                if (_highContrast)
                {
                    _spaceColor = Color.White;
                }
                else
                {
                    _spaceColor = Color.Black;
                }
            }
        }

        private static BufferedGraphicsContext _currentContext;
        private static BufferedGraphics _buffer;

        private static bool _highContrast = false;
        private static PictureBox _imageControl;
        private static float _prevScale = 0;

        private static Pen _blackHoleStroke = new Pen(Color.Red);
        private static Color _spaceColor = Color.Black;

        private static Size _prevSize = new Size();

        private static Font infoTextFont = new Font("Tahoma", 8, FontStyle.Regular);

        public static void Init(PictureBox imageControl)
        {
            _imageControl = imageControl;
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
                var bodySize = (float)body.Size;

                if (body.Visible == 1)
                {
                    using (var bodyBrush = new SolidBrush(_highContrast ? Color.Black : Color.FromArgb(body.Color)))
                    {
                        var bodyLoc = new PointF((body.LocX - body.Size * 0.5f + finalOffset.X), (body.LocY - body.Size * 0.5f + finalOffset.Y));

                        if (BodyManager.FollowSelected && body.UID == BodyManager.FollowBodyUID)
                        {
                            var followOffset = BodyManager.FollowBodyLoc();
                            RenderVars.ViewportOffset.X = -followOffset.X;
                            RenderVars.ViewportOffset.Y = -followOffset.Y;

                            //var textLoc = new PointF((float)(body.LocX - followOffset.X), (float)(body.LocY - followOffset.Y));
                            //// var textLoc = new PointF((float)(body.LocX + RenderVars.ScaleOffset.X + (body.Size * 0.5f)), (float)(body.LocY + RenderVars.ScaleOffset.Y - (body.Size * 0.5f)));

                            //OverLays.Add(new OverlayGraphic(OverlayGraphicType.Text, textLoc, string.Format("SpeedX: {0}", body.SpeedX)));
                        }

                        //Draw body.
                        _buffer.Graphics.FillEllipse(bodyBrush, bodyLoc.X, bodyLoc.Y, bodySize, bodySize);

                        // If blackhole, stroke with red circle.
                        if (body.BlackHole == 1)
                        {
                            _buffer.Graphics.DrawEllipse(_blackHoleStroke, bodyLoc.X, bodyLoc.Y, bodySize, bodySize);
                        }
                    }
                }
            }

        //    DrawOverlays();

            _buffer.Render();
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
                        _buffer.Graphics.DrawString(overlay.Value, infoTextFont, Brushes.White, overlay.Location);
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