using NBodies.Extensions;
using NBodies.Physics;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using System.Linq;

namespace NBodies.Rendering
{
    public sealed class GDIRenderer : RenderBase
    {
        private BufferedGraphicsContext _currentContext;
        private BufferedGraphics _buffer;

        private Pen _forcePen = new Pen(Color.FromArgb(150, Color.Chartreuse), 0.2f) { EndCap = LineCap.ArrowAnchor };//new Pen(Color.FromArgb(100, Color.White), 0.2f);
        private Pen _orbitPen = new Pen(Color.FromArgb(200, Color.LightGray), 0.4f) { EndCap = LineCap.ArrowAnchor };//new Pen(Color.White, 0.4f) { DashStyle = DashStyle.Dot, EndCap = LineCap.ArrowAnchor };
        private Pen _blackHoleStroke = new Pen(Color.Red);
        private SolidBrush _blurBrush = new SolidBrush(Color.FromArgb(10, Color.Black));
        private SolidBrush _statsBrush = new SolidBrush(Color.FromArgb(255, Color.Black));
        private SolidBrush _statsBackBrush = new SolidBrush(Color.FromArgb(100, Color.Black));
        private SolidBrush _bodyBrush = new SolidBrush(Color.FromArgb(255, Color.White));

        private Font _infoTextFont = new Font("Tahoma", 8, FontStyle.Regular);
        private Font _statsFont = new Font("Microsoft Sans Serif", 11, GraphicsUnit.Pixel);

        private RectangleF _statsArea = new RectangleF(0, 0, 150, 150);

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

        public override void DrawBody(Color color, float X, float Y, float size, bool isBlackHole)
        {
            X -= size * 0.5f;
            Y -= size * 0.5f;

            _bodyBrush.Color = color;

            if (!FastPrimitives)
            {
                _buffer.Graphics.FillEllipse(_bodyBrush, X, Y, size, size);
            }
            else
            {
                if (size <= 1f)
                {
                    _buffer.Graphics.FillRectangle(_bodyBrush, X, Y, size, size);
                }
                else
                {
                    _buffer.Graphics.FillEllipse(_bodyBrush, X, Y, size, size);
                }
            }

            if (isBlackHole)
            {
                _buffer.Graphics.DrawEllipse(_blackHoleStroke, X, Y, size, size);
            }
        }

        public override void DrawForceVectors(Body[] bodies, float offsetX, float offsetY)
        {
            for (int i = 0; i < bodies.Length; i++)
            {
                var body = bodies[i];

                if (!_cullTangle.Contains(body.PosX, body.PosY))
                    continue;

                var bloc = new PointF(body.PosX, body.PosY);

                var f = new PointF(body.ForceX, body.ForceY);
                f = f.Normalize();
                f = f.Multi(2f);

                var floc = bloc.Add(f);
                var finalOffset = new PointF(offsetX, offsetY);
                _buffer.Graphics.DrawLine(_forcePen, bloc.Add(finalOffset), floc.Add(finalOffset));
            }
        }

        public override void DrawMesh(MeshCell[] mesh, float offsetX, float offsetY)
        {
            if (mesh.Length < 1)
                return;

            float pSize = 0.6f;
            float pOffset = pSize / 2f;
            var meshPen = new Pen(Color.FromArgb(100, Color.Red), 0.1f);
            var pBrush = new SolidBrush(Color.FromArgb(200, Color.GreenYellow));
            Font tinyFont = new Font("Tahoma", 1, FontStyle.Regular);
            var finalOffset = new PointF(offsetX, offsetY);

            for (int i = 0; i < mesh.Length; i++)
            {
                var m = mesh[i];

                if (!_cullTangle.Contains(m.LocX, m.LocY))
                    continue;

                var meshX = m.LocX - m.Size / 2 + finalOffset.X;
                var meshY = m.LocY - m.Size / 2 + finalOffset.Y;

                _buffer.Graphics.DrawRectangle(meshPen, m.LocX - m.Size / 2 + finalOffset.X, m.LocY - m.Size / 2 + finalOffset.Y, m.Size, m.Size);

                _buffer.Graphics.FillEllipse(Brushes.Blue, m.LocX + finalOffset.X - pOffset, m.LocY + finalOffset.Y - pOffset, pSize, pSize);
                _buffer.Graphics.FillEllipse(pBrush, m.CmX + finalOffset.X - pOffset, m.CmY + finalOffset.Y - pOffset, pSize, pSize);


                //if (m.ID < PhysicsProvider.PhysicsCalc.LevelIndex[1])
                //{
                //    if ((i + 1) < mesh.Length)
                //        _buffer.Graphics.DrawLine(Pens.LimeGreen, m.Location().Add(finalOffset), mesh[i + 1].Location().Add(finalOffset));
                //}

                //  _buffer.Graphics.DrawString($@"{m.ID}", tinyFont, Brushes.White, m.LocX + finalOffset.X, m.LocY + finalOffset.Y);

                // _buffer.Graphics.DrawString($@"[{m.Level}] {m.ID}->{ m.ParentID }", tinyFont, Brushes.White, m.LocX + finalOffset.X, m.LocY + finalOffset.Y);


                //  _buffer.Graphics.DrawString($@"{m.LocX},{m.LocY}", tinyFont, Brushes.White, m.LocX + finalOffset.X, m.LocY + finalOffset.Y);
                // _buffer.Graphics.DrawString($@"{System.Math.Round(m.Mass,2)}", tinyFont, Brushes.White, m.LocX + finalOffset.X, m.LocY + finalOffset.Y);


                //_buffer.Graphics.DrawString($@"{m.xID * m.Size},{m.yID * m.Size}", tinyFont, Brushes.White, m.LocX + finalOffset.X, m.LocY + finalOffset.Y);

                //_buffer.Graphics.DrawString($@"{m.IdxX},{m.IdxY}", tinyFont, Brushes.White, m.LocX + finalOffset.X, m.LocY + finalOffset.Y);

                //  _buffer.Graphics.DrawString(BodyManager.Mesh.ToList().IndexOf(m).ToString(), tinyFont, Brushes.White, m.LocX + finalOffset.X, m.LocY + finalOffset.Y);
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

        public override void DrawOrbit(PointF[] points, PointF finalOffset)
        {
            if (points.Length < 1)
                return;

            for (int a = 0; a < points.Length; a++)
            {
                points[a] = points[a].Add(finalOffset);
            }

            _buffer.Graphics.DrawLines(_orbitPen, points);
        }

        public override void DrawBlur(Color color)
        {
            var ogSt = _buffer.Graphics.Save();
            _buffer.Graphics.ResetTransform();

            _blurBrush.Color = color;
            _buffer.Graphics.FillRectangle(_blurBrush, new RectangleF(0, 0, _viewPortSize.Width, _viewPortSize.Height));

            _buffer.Graphics.Restore(ogSt);
        }

        public override void DrawStats(string stats, System.Drawing.Color foreColor, System.Drawing.Color backColor)
        {
            var ogSt = _buffer.Graphics.Save();
            _buffer.Graphics.ResetTransform();

            if (Trails)
            {
                _statsBackBrush.Color = backColor;
                _buffer.Graphics.FillRectangle(_statsBackBrush, _statsArea);
            }

            _statsBrush.Color = foreColor;
            _buffer.Graphics.DrawString(stats, _statsFont, _statsBrush, new PointF(5, 5));

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
            base.Destroy();

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
    }
}