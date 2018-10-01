using NBodies.Structures;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Windows.Forms;
using System;
using NBodies.CUDA;
namespace NBodies.Rendering
{
    public static class Renderer
    {
        public static bool AntiAliasing = true;
        public static bool Trails = false;

        private static Bitmap _view;
        private static Graphics _gfx;
        private static PictureBox _imageControl;
        private static float _prevScale = 0;

        private static Pen _blackHoleStroke = new Pen(Color.Red);
        private static Color _spaceColor = Color.Black;

        private static Size imgSize = new Size();

        public static void Init(PictureBox imageControl)
        {
            _imageControl = imageControl;
        }

        private static void InitGfx(Size viewSize)
        {
            _view = new Bitmap(viewSize.Width, viewSize.Height, PixelFormat.Format32bppPArgb);
            imgSize = _view.Size;

            _gfx = Graphics.FromImage(_view);
        }

        public static void CheckScale()
        {
            if (_view == null || _imageControl.Size != imgSize)
            {
                InitGfx(_imageControl.Size);

                UpdateGraphicsScale();
            }

            if (_prevScale != RenderVars.CurrentScale)
            {
                UpdateGraphicsScale();
            }
        }

        private static void UpdateGraphicsScale()
        {
            _gfx.ResetTransform();
            _gfx.ScaleTransform(RenderVars.CurrentScale, RenderVars.CurrentScale);
            _prevScale = RenderVars.CurrentScale;
        }

        public static void DrawBodies(CUDAMain.Body[] bodies)
        {
            var finalOffset = PointHelper.Add(RenderVars.ViewportOffset, RenderVars.ScaleOffset);

            CheckScale();

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
                var bodySize = (float)body.Size;

                if (body.Visible == 1)
                {
                    using (var bodyBrush = new SolidBrush(Color.FromArgb(body.Color)))
                    {
                        if (BodyManager.FollowSelected)
                        {
                            var followOffset = BodyManager.FollowBodyLoc();
                            RenderVars.ViewportOffset.X = -followOffset.X;
                            RenderVars.ViewportOffset.Y = -followOffset.Y;
                        }
                       
                        //Draw body.
                        var bodyLoc = new PointF((float)(body.LocX - body.Size * 0.5f + finalOffset.X), (float)(body.LocY - body.Size * 0.5f + finalOffset.Y));
                        _gfx.FillEllipse(bodyBrush, bodyLoc.X, bodyLoc.Y, bodySize, bodySize);

                        // If blackhole, stroke with red circle.
                        if (body.BlackHole == 1)
                        {
                            _gfx.DrawEllipse(_blackHoleStroke, bodyLoc.X, bodyLoc.Y, bodySize, bodySize);
                        }
                    }
                }
            }

            SetControlImage(_view);
        }

        private static void SetControlImage(Bitmap image)
        {
            if (_imageControl.InvokeRequired)
            {
                var asyncResult = _imageControl.BeginInvoke(new Action(() => SetControlImage(image)));
                asyncResult.AsyncWaitHandle.WaitOne(MainLoop.MinFrameTime);
                // _imageControl.Invoke(new Action(() => SetControlImage(image)));
            }
            else
            {
                _imageControl.Image = image;
                _imageControl.Invalidate();
            }
        }
    }
}