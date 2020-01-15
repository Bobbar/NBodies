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
        private Matrix3x2 _scaleTransform = new Matrix3x2();
        private Matrix3x2 _defaultScaleTransform = new Matrix3x2(1, 0, 0, 1, 0, 0);
        private SharpDX.RectangleF _statsArea = new SharpDX.RectangleF(5, 5, 150, 150);

        private d2.SolidColorBrush _bodyBrush;
        private d2.SolidColorBrush _whiteBrush;
        private d2.SolidColorBrush _greenBrush;
        private d2.SolidColorBrush _grayBrush;
        private d2.SolidColorBrush _orbitBrush;
        private d2.SolidColorBrush _redBrush;
        private d2.SolidColorBrush _forceBrush;
        private d2.SolidColorBrush _blurBrush;
        private d2.SolidColorBrush _statsBrush;
        private d2.SolidColorBrush _statsBackBrush;

        private d2.SolidColorBrush _meshBrush;
        private d2.SolidColorBrush _centerBrush;
        private d2.SolidColorBrush _massBrush;

        private dw.TextFormat _statsFont;

        private d2.Ellipse _bodyEllipse;
        private SharpDX.RectangleF _bodyRect;
        private SharpDX.RectangleF _blurRect;
        private RawColor4 _bodyColor;

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
            _bodyRect = new SharpDX.RectangleF(0, 0, 0, 0);
            _blurRect = new SharpDX.RectangleF(0, 0, 0, 0);
            _blurBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.FromArgb(10, System.Drawing.Color.Black)));
            _infoText = new dw.TextFormat(_dwFact, "Tahoma", dw.FontWeight.Normal, dw.FontStyle.Normal, 11);
            _whiteBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.White));
            _greenBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.LimeGreen));
            _orbitBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.FromArgb(200, System.Drawing.Color.LightGray)));
            _grayBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.FromArgb(200, System.Drawing.Color.LightGray)));
            _redBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.Red));
            _forceBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.FromArgb(100, System.Drawing.Color.Chartreuse)));
            _meshBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.FromArgb(200, System.Drawing.Color.Red)));
            _centerBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.Blue));
            _massBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.FromArgb(200, System.Drawing.Color.GreenYellow)));
            _statsBrush = new d2.SolidColorBrush(_wndRender, new RawColor4(255, 0, 0, 0));
            _statsBackBrush = new d2.SolidColorBrush(_wndRender, ConvertColor(System.Drawing.Color.FromArgb(100, System.Drawing.Color.Black)));
            _statsFont = new dw.TextFormat(_dwFact, "Microsoft Sans Serif", 11);

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

        public override void DrawBodiesRaw(Body[] bodies) { }

        public override void DrawBody(System.Drawing.Color color, float X, float Y, float size, bool isBlackHole)
        {
            _bodyColor.A = color.A / 255f;
            _bodyColor.R = color.R / 255f;
            _bodyColor.G = color.G / 255f;
            _bodyColor.B = color.B / 255f;

            _bodyBrush.Color = _bodyColor;

            _bodyEllipse.Point.X = X;
            _bodyEllipse.Point.Y = Y;
            _bodyEllipse.RadiusX = size * 0.5f;
            _bodyEllipse.RadiusY = size * 0.5f;

            float offset = size * 0.5f;
            _bodyRect.X = X - offset;
            _bodyRect.Y = Y - offset;
            _bodyRect.Width = size;
            _bodyRect.Height = size;

            if (!FastPrimitives)
            {
                _wndRender.FillEllipse(_bodyEllipse, _bodyBrush);
            }
            else
            {
                if (size <= 1f)
                {
                    _wndRender.FillRectangle(_bodyRect, _bodyBrush);
                }
                else
                {
                    _wndRender.FillEllipse(_bodyEllipse, _bodyBrush);
                }
            }

            if (isBlackHole)
            {
                _wndRender.DrawEllipse(_bodyEllipse, _redBrush);
            }
        }

        public override void DrawForceVectors(Body[] bodies, float offsetX, float offsetY)
        {
            var finalOffset = new PointF(offsetX, offsetY);

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

                _wndRender.DrawLine(bloc.Add(finalOffset).ToVector(), floc.Add(finalOffset).ToVector(), _forceBrush, 0.2f, _arrowStyle);
            }
        }

        public override void DrawMesh(MeshCell[] mesh, float offsetX, float offsetY)
        {
            float pSize = 0.3f;
            float pOffset = pSize / 2f;

            foreach (var m in mesh)
            {
                float pSizeLvl = pSize + m.Level * 0.2f;

                if (_cullTangle.Contains(m.LocX, m.LocY))
                {
                    var meshX = m.LocX - m.Size / 2 + offsetX;
                    var meshY = m.LocY - m.Size / 2 + offsetY;
                    _wndRender.DrawRectangle(new SharpDX.RectangleF(meshX, meshY, m.Size, m.Size), _meshBrush, 0.2f);
                }

                if (_cullTangle.Contains(m.CmX, m.CmY))
                {
                    var massEllip = new d2.Ellipse(new Vector2(m.CmX + offsetX, m.CmY + offsetY), pSizeLvl, pSizeLvl);
                    _wndRender.FillEllipse(massEllip, _massBrush);
                }

                //_buffer.Graphics.DrawString($@"{m.xID},{m.yID}", tinyFont, Brushes.White, m.LocX + finalOffset.X, m.LocY + finalOffset.Y);
                //_buffer.Graphics.DrawString(BodyManager.Mesh.ToList().IndexOf(m).ToString(), _infoTextFont, Brushes.White, m.LocX + finalOffset.X, m.LocY + finalOffset.Y);
            }


            //// Draws origin.
            //_wndRender.DrawLine(new Vector2(0 + offsetX, -5000 + offsetY), new Vector2(0 + offsetX, 5000 + offsetY), _greenBrush, 0.2f);
            //_wndRender.DrawLine(new Vector2(-5000 + offsetX, 0 + offsetY), new Vector2(5000 + offsetX, 0 + offsetY), _greenBrush, 0.2f);

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
            _wndRender.Transform = _defaultScaleTransform;

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

        public override void DrawBlur(System.Drawing.Color color)
        {
            var ogSt = _wndRender.Transform;
            _wndRender.Transform = _defaultScaleTransform;

            _blurBrush.Color = ConvertColor(color);
            _blurRect.X = 0;
            _blurRect.Y = 0;
            _blurRect.Size = new Size2F(_viewPortSize.Width, _viewPortSize.Height);

            _wndRender.FillRectangle(_blurRect, _blurBrush);

            _wndRender.Transform = ogSt;
        }

        public override void DrawStats(string stats, System.Drawing.Color foreColor, System.Drawing.Color backColor)
        {
            var ogSt = _wndRender.Transform;
            _wndRender.Transform = _defaultScaleTransform;

            if (Trails)
            {
                _statsBackBrush.Color = ConvertColor(backColor);
                _wndRender.FillRectangle(_statsArea, _statsBackBrush);
            }

            _statsBrush.Color = ConvertColor(foreColor);
            _wndRender.DrawText(stats, _statsFont, _statsArea, _statsBrush, SharpDX.Direct2D1.DrawTextOptions.None);

            _wndRender.Transform = ogSt;
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
            _wndRender.Dispose();

            _fact.Dispose();
            _dwFact.Dispose();

            _bodyBrush.Dispose();
            _whiteBrush.Dispose();
            _greenBrush.Dispose();
            _grayBrush.Dispose();
            _orbitBrush.Dispose();
            _redBrush.Dispose();
            _meshBrush.Dispose();
            _centerBrush.Dispose();
            _massBrush.Dispose();
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
            _scaleTransform.ScaleVector = new Vector2(currentScale, currentScale);
            _wndRender.Transform = _scaleTransform;
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