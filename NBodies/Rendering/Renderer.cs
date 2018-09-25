using NBodies.Structures;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Windows.Forms;

namespace NBodies.Rendering
{
    public static class Renderer
    {
        public static bool AntiAliasing = true;
        public static bool Trails = false;

        private static Bitmap _view;
        private static Graphics _gfx;
        private static PictureBox _imageControl;
        private static float _prevScale;

        private static Pen _blackHoleStroke = new Pen(Color.Red);
        private static Color _spaceColor = Color.Black;

        public static void Init(Size viewSize, PictureBox imageControl)
        {
            _imageControl = imageControl;

            InitGfx(imageControl.Size);
        }

        private static void InitGfx(Size viewSize)
        {
            _view = new Bitmap(viewSize.Width, viewSize.Height, PixelFormat.Format32bppPArgb);
            _gfx = Graphics.FromImage(_view);
        }

        public static void UpdateScale()
        {
            if (_imageControl.Size != _view.Size)
            {
                InitGfx(_imageControl.Size);
            }

            if (_prevScale != RenderVars.CurrentScale)
            {
                _gfx.ResetTransform();
                _gfx.ScaleTransform(RenderVars.CurrentScale, RenderVars.CurrentScale);
            }
        }

        public static void DrawBodies(Body[] bodies)
        {
            PointF viewOffset = new PointF();

            UpdateScale();

            if (!Trails) _gfx.Clear(_spaceColor);

            if (AntiAliasing)
            {
                _gfx.SmoothingMode = SmoothingMode.HighQuality;
            }
            else
            {
                _gfx.SmoothingMode = SmoothingMode.None;
            }

            for (int i = 0; i < bodies.Length; i++)
            {
                var body = bodies[i];
                var bodyLoc = new PointF((float)(body.LocX - body.Size * 0.5f + FinalOffset().X), (float)(body.LocY - body.Size * 0.5f + FinalOffset().Y));
                var bodySize = (float)body.Size;

                if (body.Visible == 1)
                {
                    using (var bodyBrush = new SolidBrush(Color.FromArgb(body.Color)))
                    {
                        if (BodyManager.FollowSelected)
                        {
                            var followOffset = BodyManager.FollowBodyLoc();
                            viewOffset.X = -followOffset.X;
                            viewOffset.Y = -followOffset.Y;
                        }
                        else
                        {
                            viewOffset = RenderVars.ViewportOffset;
                        }

                        //Draw body.
                        _gfx.FillEllipse(bodyBrush, bodyLoc.X, bodyLoc.Y, bodySize, bodySize);

                        // If blackhole, stroke with red circle.
                        if (body.BlackHole == 1)
                        {
                            _gfx.DrawEllipse(_blackHoleStroke, bodyLoc.X, bodyLoc.Y, bodySize, bodySize);
                        }
                    }
                }
            }

            _imageControl.Image = _view;
            _imageControl.Invalidate();
        }

        private static PointF FinalOffset()
        {
            return new PointF(RenderVars.ViewportOffset.X + RenderVars.ScaleOffset.X, RenderVars.ViewportOffset.Y + RenderVars.ScaleOffset.Y);
        }
    }
}