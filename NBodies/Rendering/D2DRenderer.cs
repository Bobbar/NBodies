using NBodies.Extensions;
using NBodies.Physics;
using SharpDX;
using SharpDX.Mathematics.Interop;
using System.Drawing;
using System.Windows.Forms;
using d2 = SharpDX.Direct2D1;
using dw = SharpDX.DirectWrite;
using dxgi = SharpDX.DXGI;
using System;

namespace NBodies.Rendering
{
    public sealed class D2DRenderer : RenderBase
    {
        private d2.Factory _fact = new d2.Factory(d2.FactoryType.MultiThreaded);
        private dw.Factory _dwFact = new dw.Factory(dw.FactoryType.Shared);
        private d2.HwndRenderTargetProperties _hwndProperties;
        private d2.RenderTargetProperties _rndTargProperties;
        private d2.WindowRenderTarget _wndRender;
        private Matrix3x2 _transform = new Matrix3x2();

        private d2.SolidColorBrush _bodyBrush;
        private d2.SolidColorBrush _whiteBrush;
        private d2.SolidColorBrush _greenBrush;
        private d2.SolidColorBrush _grayBrush;
        private d2.SolidColorBrush _orbitBrush;
        private d2.SolidColorBrush _redBrush;
        private d2.SolidColorBrush _forceBrush;

        private d2.Ellipse _bodyEllipse;
        private dw.TextFormat _infoText;
        private d2.StrokeStyle _arrowStyle;

        public D2DRenderer(Control targetControl) : base(targetControl)
        {
        }

        public override string ToString()
        {
            return "Direct 2D";
        }

        public override void Clear(System.Drawing.Color color)
        {
            _wndRender.Clear(ConvertColor(color));
        }

        public override void InitGraphics()
        {
            _rndTargProperties = new d2.RenderTargetProperties(new d2.PixelFormat(dxgi.Format.B8G8R8A8_UNorm, d2.AlphaMode.Premultiplied));
            //_rndTargProperties.MinLevel = d2.FeatureLevel.Level_10;
            // _rndTargProperties.Usage = d2.RenderTargetUsage.GdiCompatible;

            InitProperties(_targetControl);

            _wndRender = new d2.WindowRenderTarget(_fact, _rndTargProperties, _hwndProperties);

            _bodyBrush = new d2.SolidColorBrush(_wndRender, new Color4(0, 0, 0, 0));
            _bodyEllipse = new d2.Ellipse(new Vector2(), 0, 0);
            _infoText = new dw.TextFormat(_dwFact, "Tahoma", dw.FontWeight.Normal, dw.FontStyle.Normal, 11);
            _whiteBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.White));
            _greenBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.LimeGreen));
            _orbitBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.FromArgb(200, System.Drawing.Color.LightGray)));
            _grayBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.FromArgb(200, System.Drawing.Color.LightGray)));
            _redBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.Red));
            _forceBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.FromArgb(100, System.Drawing.Color.Chartreuse)));

            var arrowProps = new d2.StrokeStyleProperties() { EndCap = d2.CapStyle.Triangle };
            _arrowStyle = new d2.StrokeStyle(_fact, arrowProps);

            _viewPortSize = _targetControl.Size;
        }

        private void InitProperties(Control targetControl)
        {
            if (targetControl.InvokeRequired)
            {
                targetControl.Invoke(new Action(() => InitProperties(targetControl)));
            }
            else
            {
                _hwndProperties = new d2.HwndRenderTargetProperties();
                _hwndProperties.Hwnd = _targetControl.Handle;
                _hwndProperties.PixelSize = new Size2(targetControl.ClientSize.Width, targetControl.ClientSize.Height);
                _hwndProperties.PresentOptions = d2.PresentOptions.Immediately;
            }
        }

        public override void DrawBody(Body body, System.Drawing.Color color, float X, float Y, float size)
        {
            _bodyBrush.Color = ConvertColor(color);
            _bodyEllipse.Point.X = X;
            _bodyEllipse.Point.Y = Y;
            _bodyEllipse.RadiusX = body.Size * 0.5f;
            _bodyEllipse.RadiusY = body.Size * 0.5f;

            _wndRender.FillEllipse(_bodyEllipse, _bodyBrush);

            if (body.BlackHole == 1)
            {
                _wndRender.DrawEllipse(_bodyEllipse, _redBrush);

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
                _wndRender.DrawLine(bloc.Add(finalOffset).ToVector(), floc.Add(finalOffset).ToVector(), _forceBrush, 0.2f, _arrowStyle);
            }
        }

        public override void DrawMesh(MeshCell[] mesh, float offsetX, float offsetY)
        {
            float pSize = 0.3f;
            float pOffset = pSize / 2f;
            var meshBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.Red));
            var centerBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.Blue));
            var massBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.FromArgb(200, System.Drawing.Color.GreenYellow)));

            foreach (var m in mesh)
            {
                if (!_cullTangle.Contains(m.LocX, m.LocY))
                    continue;

                var meshX = m.LocX - m.Size / 2 + offsetX;
                var meshY = m.LocY - m.Size / 2 + offsetY;

                _wndRender.DrawRectangle(new SharpDX.RectangleF(meshX, meshY, m.Size, m.Size), meshBrush, 0.1f);

                var centerEllip = new d2.Ellipse(new Vector2(m.LocX + offsetX, m.LocY + offsetY), pSize, pSize);

                _wndRender.FillEllipse(centerEllip, centerBrush);

                var massEllip = new d2.Ellipse(new Vector2(m.CmX + offsetX, m.CmY + offsetY), pSize, pSize);

                _wndRender.FillEllipse(massEllip, massBrush);

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

            var ogSt = _wndRender.Transform;
            _wndRender.Transform = new Matrix3x2(1, 0, 0, 1, 0, 0);
            foreach (var overlay in OverLays.ToArray())
            {
                if (overlay.Visible)
                {
                    switch (overlay.Type)
                    {
                        case OverlayGraphicType.Text:
                            _wndRender.DrawText(overlay.Value, _infoText, new SharpDX.RectangleF(overlay.Location.X, overlay.Location.Y, 200f, 200f), _whiteBrush);
                            break;

                        case OverlayGraphicType.Line:
                            _wndRender.DrawLine(overlay.Location.ToVector(), overlay.Location2.ToVector(), _greenBrush, 5.0f, _arrowStyle);
                            break;
                    }
                }
            }

            _wndRender.Transform = ogSt;
        }

        public override void DrawOrbit(PointF[] points, PointF finalOffset)
        {
            if (points.Length < 1)
                return;

            for (int a = 1; a < points.Length; a++)
            {
                _wndRender.DrawLine(points[a - 1].Add(finalOffset).ToVector(), points[a].Add(finalOffset).ToVector(), _orbitBrush, 0.4f, _arrowStyle);
            }
        }

        public override void BeginDraw()
        {
            _wndRender.BeginDraw();
        }

        public override void EndDraw()
        {
            _wndRender.EndDraw();
        }

        public override void Destroy()
        {
            _fact.Dispose();
            _dwFact.Dispose();
          //  _wndRender.Flush();
            _wndRender.Dispose();

            _bodyBrush.Dispose();
            _whiteBrush.Dispose();
            _greenBrush.Dispose();
            _grayBrush.Dispose();
            _orbitBrush.Dispose();
            _redBrush.Dispose();
        }

        public override void SetAntiAliasing(bool enabled)
        {
            if (enabled)
            {
                _wndRender.AntialiasMode = d2.AntialiasMode.PerPrimitive;
            }
            else
            {
                _wndRender.AntialiasMode = d2.AntialiasMode.Aliased;
            }
        }

        public override void UpdateGraphicsScale(float currentScale)
        {
            _transform.ScaleVector = new Vector2(currentScale, currentScale);
            _wndRender.Transform = _transform;
            _prevScale = currentScale;
        }

        public override void UpdateViewportSize(float width, float height)
        {
            //InitGraphics();
            _wndRender.Resize(new Size2((int)width, (int)height));
        }

        private RawColor4 ConvertColor(System.Drawing.Color color)
        {
            return new RawColor4(color.R / 255f, color.G / 255f, color.B / 255f, color.A / 255f);
        }

      
    }
}