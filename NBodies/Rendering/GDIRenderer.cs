using NBodies.Extensions;
using NBodies.Physics;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using System;

namespace NBodies.Rendering
{
    public sealed class GDIRenderer : RenderBase
    {
        private BufferedGraphicsContext _currentContext;
        private BufferedGraphics _buffer;
        private Dictionary<int, SolidBrush> _brushCache = new Dictionary<int, SolidBrush>();

        private Pen _forcePen = new Pen(Color.FromArgb(150, Color.LightGray), 0.2f) { EndCap = LineCap.ArrowAnchor };//new Pen(Color.FromArgb(100, Color.White), 0.2f);
        private Pen _orbitPen = new Pen(Color.FromArgb(200, Color.LightGray), 0.4f) { EndCap = LineCap.ArrowAnchor };//new Pen(Color.White, 0.4f) { DashStyle = DashStyle.Dot, EndCap = LineCap.ArrowAnchor };
        private Pen _blackHoleStroke = new Pen(Color.Red);

        private Font _infoTextFont = new Font("Tahoma", 8, FontStyle.Regular);

        public GDIRenderer(Control targetControl) : base(targetControl)
        {
        }

        public override string ToString()
        {
            return "GDI";
        }

        public override void InitGraphics()
        {
            _currentContext = BufferedGraphicsManager.Current;
            _buffer = _currentContext.Allocate(_targetControl.CreateGraphics(), _targetControl.DisplayRectangle);
           
            //_buffer.Graphics.CompositingMode = CompositingMode.SourceOver;
            //_buffer.Graphics.CompositingQuality = CompositingQuality.HighSpeed;
            //_buffer.Graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
            //_buffer.Graphics.PixelOffsetMode = PixelOffsetMode.HighSpeed;

            _viewPortSize = _targetControl.Size;
        }

        public override void Clear(Color color)
        {
            _buffer.Graphics.Clear(color);
        }

        public override void DrawBody(Body body, Color color, float X, float Y, float size)
        {
            X -= size * 0.5f;
            Y -= size * 0.5f;

            int brushID = color.ToArgb();

            if (!_brushCache.ContainsKey(brushID))
            {
                _brushCache.Add(brushID, new SolidBrush(color));
            }

            var bodyBrush = _brushCache[brushID];

            _buffer.Graphics.FillEllipse(bodyBrush, X, Y, size, size);

            if (body.BlackHole == 1)
            {
                _buffer.Graphics.DrawEllipse(_blackHoleStroke, X, Y, size, size);
            }
        }

        public override void DrawForceVectors(Body[] bodies, float offsetX, float offsetY)
        {
            for (int i = 0; i < bodies.Length; i++)
            {
                var body = bodies[i];

                if (!_cullTangle.Contains(body.LocX, body.LocY))
                    continue;

                var bloc = new PointF(body.LocX, body.LocY);

                var f = new PointF(body.ForceX, body.ForceY);
                f = f.Div(f.LengthSqrt());

                //f = f.Multi(0.01f);
                var floc = bloc.Add(f);
                var finalOffset = new PointF(offsetX, offsetY);
                _buffer.Graphics.DrawLine(_forcePen, bloc.Add(finalOffset), floc.Add(finalOffset));
            }
        }

        public override void DrawMesh(MeshCell[] mesh, float offsetX, float offsetY)
        {
            float pSize = 1.0f;
            float pOffset = pSize / 2f;
            var meshPen = new Pen(Color.FromArgb(100, Color.Red), 0.1f);
            var pBrush = new SolidBrush(Color.FromArgb(200, Color.GreenYellow));
            Font tinyFont = new Font("Tahoma", 3, FontStyle.Regular);
            var finalOffset = new PointF(offsetX, offsetY);

            foreach (var m in mesh)
            {
                if (!_cullTangle.Contains(m.LocX, m.LocY))
                    continue;

                var meshX = m.LocX - m.Size / 2 + finalOffset.X;
                var meshY = m.LocY - m.Size / 2 + finalOffset.Y;

                _buffer.Graphics.DrawRectangle(meshPen, m.LocX - m.Size / 2 + finalOffset.X, m.LocY - m.Size / 2 + finalOffset.Y, m.Size, m.Size);

                _buffer.Graphics.FillEllipse(Brushes.Blue, m.LocX + finalOffset.X - pOffset, m.LocY + finalOffset.Y - pOffset, pSize, pSize);
                _buffer.Graphics.FillEllipse(pBrush, m.CmX + finalOffset.X - pOffset, m.CmY + finalOffset.Y - pOffset, pSize, pSize);

                //_buffer.Graphics.DrawString($@"{m.xID},{m.yID}", tinyFont, Brushes.White, m.LocX + finalOffset.X, m.LocY + finalOffset.Y);

                //_buffer.Graphics.DrawString(BodyManager.Mesh.ToList().IndexOf(m).ToString(), _infoTextFont, Brushes.White, m.LocX + finalOffset.X, m.LocY + finalOffset.Y);
            }
        }

        public override void DrawOverlays(float offsetX, float offsetY)
        {
            var finalOffset = new PointF(offsetX, offsetY);

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
                            _buffer.Graphics.DrawLine(new Pen(Color.LimeGreen) { Width = 5.0f, EndCap = LineCap.ArrowAnchor }, overlay.Location, overlay.Location2);
                            break;
                    }
                }
            }

            _buffer.Graphics.Restore(ogSt);
        }

        public override void BeginDraw()
        {
            // Nothing to do.
        }

        public override void EndDraw()
        {
            if (!_targetControl.IsDisposed && !_targetControl.Disposing)
                _buffer.Render();
        }

        public override void Destroy()
        {
            _buffer.Dispose();
            _currentContext.Dispose();
        }

        public override void SetAntiAliasing(bool enabled)
        {
            if (enabled)
            {
                _buffer.Graphics.SmoothingMode = SmoothingMode.HighQuality;
            }
            else
            {
                _buffer.Graphics.SmoothingMode = SmoothingMode.None;
            }
        }

        public override void UpdateGraphicsScale(float currentScale)
        {
            _buffer.Graphics.ResetTransform();
            _buffer.Graphics.ScaleTransform(currentScale, currentScale);
            _prevScale = currentScale;
        }

        public override void UpdateViewportSize(float width, float height)
        {
            InitGraphics();
        }

        private void DrawOrbit(PointF[] points, PointF finalOffset)
        {
            if (points.Length < 1)
                return;

            for (int a = 0; a < points.Length; a++)
            {
                points[a] = points[a].Add(finalOffset);
            }

            _buffer.Graphics.DrawLines(_orbitPen, points);
        }

       
    }
}