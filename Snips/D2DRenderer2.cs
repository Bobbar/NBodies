using NBodies.Extensions;
using NBodies.Physics;
using SharpDX;
using SharpDX.Mathematics.Interop;
using System.Drawing;
using System.Windows.Forms;
using d2 = SharpDX.Direct2D1;
using dw = SharpDX.DirectWrite;
using dxgi = SharpDX.DXGI;
using d11 = SharpDX.Direct3D11;
using d3d = SharpDX.Direct3D;
using System;

namespace NBodies.Rendering
{
    public sealed class D2DRenderer2 : RenderBase
    {
        //d11.Device1 device;
        d11.Device device;

        //d11.DeviceContext1 context;


        //d2.Device d2dDevice;
        //d2.DeviceContext d2dContext;


        //d2.BitmapProperties1 properties;

        //dxgi.Surface backBuffer;
        //dxgi.SwapChain1 swapChain;
        dxgi.SwapChain swapChain;


        //  dxgi.Factory2 dxgiFactory2;

        // d2.Bitmap d2dTarget;

        d2.RenderTarget d2dRenderTarget;

        private d2.Factory _fact;// = new d2.Factory(d2.FactoryType.MultiThreaded);
        private dw.Factory _dwFact;// = new dw.Factory(dw.FactoryType.Shared);
        //private d2.HwndRenderTargetProperties _hwndProperties;
        //private d2.RenderTargetProperties _rndTargProperties;
        //private d2.WindowRenderTarget _wndRender;
        private Matrix3x2 _transform = new Matrix3x2();

        private d2.SolidColorBrush _bodyBrush;
        private d2.SolidColorBrush _whiteBrush;
        private d2.SolidColorBrush _greenBrush;
        private d2.SolidColorBrush _grayBrush;
        private d2.SolidColorBrush _orbitBrush;
        private d2.SolidColorBrush _redBrush;
        private d2.SolidColorBrush _forceBrush;

        private d2.SolidColorBrush _meshBrush;
        private d2.SolidColorBrush _centerBrush;
        private d2.SolidColorBrush _massBrush;

        private d2.Ellipse _bodyEllipse;
        private dw.TextFormat _infoText;
        private d2.StrokeStyle _arrowStyle;

        public D2DRenderer2(Control targetControl) : base(targetControl)
        {
        }

        public override string ToString()
        {
            return "Direct 2D";
        }

        public override void Clear(System.Drawing.Color color)
        {
            d2dRenderTarget.Clear(ConvertColor(color));
        }

        public override void InitGraphics()
        {
            //d11.Device defaultDevice = new d11.Device(SharpDX.Direct3D.DriverType.Hardware, d11.DeviceCreationFlags.None);

            //device = defaultDevice.QueryInterface<d11.Device1>();
            //context = device.ImmediateContext.QueryInterface<d11.DeviceContext1>();

            //dxgi.Device2 dxgiDevice2 = device.QueryInterface<dxgi.Device2>();
            //dxgi.Adapter dxgiAdapter = dxgiDevice2.Adapter;
            //dxgiFactory2 = dxgiAdapter.GetParent<dxgi.Factory2>();

            //var cprop = new d2.CreationProperties();
            //cprop.Options = d2.DeviceContextOptions.None;
            //cprop.DebugLevel = d2.DebugLevel.Warning;


            //d2dDevice = new SharpDX.Direct2D1.Device(dxgiDevice2, cprop);
            //d2dContext = new SharpDX.Direct2D1.DeviceContext(d2dDevice, d2.DeviceContextOptions.None);


            _fact = new d2.Factory(d2.FactoryType.MultiThreaded);
            _dwFact = new dw.Factory(dw.FactoryType.Shared);

            //_rndTargProperties = new d2.RenderTargetProperties(new d2.PixelFormat(dxgi.Format.B8G8R8A8_UNorm, d2.AlphaMode.Premultiplied));
            //_rndTargProperties.MinLevel = d2.FeatureLevel.Level_10;
            // _rndTargProperties.Usage = d2.RenderTargetUsage.GdiCompatible;

            InitProperties(_targetControl);

            //  _wndRender = new d2.WindowRenderTarget(_fact, _rndTargProperties, _hwndProperties);


            _bodyBrush = new d2.SolidColorBrush(d2dRenderTarget, new Color4(0, 0, 0, 0));
            _bodyEllipse = new d2.Ellipse(new Vector2(), 0, 0);
            _infoText = new dw.TextFormat(_dwFact, "Tahoma", dw.FontWeight.Normal, dw.FontStyle.Normal, 11);
            _whiteBrush = new d2.SolidColorBrush(d2dRenderTarget, ConvertColor(System.Drawing.Color.White));
            _greenBrush = new d2.SolidColorBrush(d2dRenderTarget, ConvertColor(System.Drawing.Color.LimeGreen));
            _orbitBrush = new d2.SolidColorBrush(d2dRenderTarget, ConvertColor(System.Drawing.Color.FromArgb(200, System.Drawing.Color.LightGray)));
            _grayBrush = new d2.SolidColorBrush(d2dRenderTarget, ConvertColor(System.Drawing.Color.FromArgb(200, System.Drawing.Color.LightGray)));
            _redBrush = new d2.SolidColorBrush(d2dRenderTarget, ConvertColor(System.Drawing.Color.Red));
            _forceBrush = new d2.SolidColorBrush(d2dRenderTarget, ConvertColor(System.Drawing.Color.FromArgb(100, System.Drawing.Color.Chartreuse)));
            _meshBrush = new d2.SolidColorBrush(d2dRenderTarget, ConvertColor(System.Drawing.Color.FromArgb(200, System.Drawing.Color.Red)));
            _centerBrush = new d2.SolidColorBrush(d2dRenderTarget, ConvertColor(System.Drawing.Color.Blue));
            _massBrush = new d2.SolidColorBrush(d2dRenderTarget, ConvertColor(System.Drawing.Color.FromArgb(200, System.Drawing.Color.GreenYellow)));

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

                var desc = new dxgi.SwapChainDescription()
                {
                    BufferCount = 2,
                    ModeDescription =
                           new dxgi.ModeDescription(targetControl.ClientSize.Width, targetControl.ClientSize.Height,
                                               new dxgi.Rational(60, 1), dxgi.Format.R8G8B8A8_UNorm),
                    IsWindowed = true,
                    OutputHandle = targetControl.Handle,
                    SampleDescription = new dxgi.SampleDescription(1, 0),
                    SwapEffect = dxgi.SwapEffect.FlipSequential,
                    Usage = dxgi.Usage.RenderTargetOutput
                };

                d11.Device.CreateWithSwapChain(d3d.DriverType.Hardware, d11.DeviceCreationFlags.BgraSupport, new SharpDX.Direct3D.FeatureLevel[] { SharpDX.Direct3D.FeatureLevel.Level_11_0 }, desc, out device, out swapChain);

                // var d2dFactory = new SharpDX.Direct2D1.Factory();


                d11.Texture2D backBuffer = d11.Texture2D.FromSwapChain<d11.Texture2D>(swapChain, 0);


                dxgi.Surface surface = backBuffer.QueryInterface<dxgi.Surface>();

                //d2dRenderTarget = new d2.RenderTarget(d2dFactory, surface, new d2.RenderTargetProperties(new d2.PixelFormat(dxgi.Format.Unknown, d2.AlphaMode.Premultiplied)));
                d2dRenderTarget = new d2.RenderTarget(_fact, surface, new d2.RenderTargetProperties(new d2.PixelFormat(dxgi.Format.Unknown, d2.AlphaMode.Premultiplied)));

                //_hwndProperties = new d2.HwndRenderTargetProperties();
                //_hwndProperties.Hwnd = _targetControl.Handle;
                //_hwndProperties.PixelSize = new Size2(targetControl.ClientSize.Width, targetControl.ClientSize.Height);
                //_hwndProperties.PresentOptions = d2.PresentOptions.Immediately;
            }
        }

        public override void DrawBody(Body body, System.Drawing.Color color, float X, float Y, float size)
        {
            _bodyBrush.Color = ConvertColor(color);
            _bodyEllipse.Point.X = X;
            _bodyEllipse.Point.Y = Y;
            _bodyEllipse.RadiusX = body.Size * 0.5f;
            _bodyEllipse.RadiusY = body.Size * 0.5f;

            d2dRenderTarget.FillEllipse(_bodyEllipse, _bodyBrush);

            if (body.BlackHole == 1)
            {
                d2dRenderTarget.DrawEllipse(_bodyEllipse, _redBrush);

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
                f = f.Normalize();
                f = f.Multi(2f);

                var floc = bloc.Add(f);
                var finalOffset = new PointF(offsetX, offsetY);
                d2dRenderTarget.DrawLine(bloc.Add(finalOffset).ToVector(), floc.Add(finalOffset).ToVector(), _forceBrush, 0.2f, _arrowStyle);
            }
        }

        public override void DrawMesh(MeshCell[] mesh, float offsetX, float offsetY)
        {
            float pSize = 0.3f;
            float pOffset = pSize / 2f;

            foreach (var m in mesh)
            {
                if (!_cullTangle.Contains(m.LocX, m.LocY))
                    continue;

                var meshX = m.LocX - m.Size / 2 + offsetX;
                var meshY = m.LocY - m.Size / 2 + offsetY;

                d2dRenderTarget.DrawRectangle(new SharpDX.RectangleF(meshX, meshY, m.Size, m.Size), _meshBrush, 0.2f);

                var centerEllip = new d2.Ellipse(new Vector2(m.LocX + offsetX, m.LocY + offsetY), pSize, pSize);

                d2dRenderTarget.FillEllipse(centerEllip, _centerBrush);

                var massEllip = new d2.Ellipse(new Vector2(m.CmX + offsetX, m.CmY + offsetY), pSize, pSize);

                d2dRenderTarget.FillEllipse(massEllip, _massBrush);

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

            var ogSt = d2dRenderTarget.Transform;
            d2dRenderTarget.Transform = new Matrix3x2(1, 0, 0, 1, 0, 0);
            foreach (var overlay in OverLays.ToArray())
            {
                if (overlay.Visible)
                {
                    switch (overlay.Type)
                    {
                        case OverlayGraphicType.Text:
                            d2dRenderTarget.DrawText(overlay.Value, _infoText, new SharpDX.RectangleF(overlay.Location.X, overlay.Location.Y, 200f, 200f), _whiteBrush);
                            break;

                        case OverlayGraphicType.Line:
                            d2dRenderTarget.DrawLine(overlay.Location.ToVector(), overlay.Location2.ToVector(), _greenBrush, 5.0f, _arrowStyle);
                            break;
                    }
                }
            }

            d2dRenderTarget.Transform = ogSt;
        }

        public override void DrawOrbit(PointF[] points, PointF finalOffset)
        {
            if (points.Length < 1)
                return;

            for (int a = 1; a < points.Length; a++)
            {
                d2dRenderTarget.DrawLine(points[a - 1].Add(finalOffset).ToVector(), points[a].Add(finalOffset).ToVector(), _orbitBrush, 0.4f, _arrowStyle);
            }
        }

        public override void BeginDraw()
        {
            d2dRenderTarget.BeginDraw();
        }

        public override void EndDraw()
        {
            d2dRenderTarget.EndDraw();
            swapChain.Present(0, dxgi.PresentFlags.None);
        }

        public override void Destroy()
        {
            d2dRenderTarget.Dispose();

            device.Dispose();
            swapChain.Dispose();

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
                d2dRenderTarget.AntialiasMode = d2.AntialiasMode.PerPrimitive;
            }
            else
            {
                d2dRenderTarget.AntialiasMode = d2.AntialiasMode.Aliased;
            }
        }

        public override void UpdateGraphicsScale(float currentScale)
        {
            _transform.ScaleVector = new Vector2(currentScale, currentScale);
            d2dRenderTarget.Transform = _transform;
            _prevScale = currentScale;
        }

        public override void UpdateViewportSize(float width, float height)
        {
            Destroy();
            InitGraphics();
            //InitGraphics();
            //_wndRender.Resize(new Size2((int)width, (int)height));
            // swapChain.ResizeBuffers(1, (int)width, (int)height, dxgi.Format.Unknown, dxgi.SwapChainFlags.None);
        }

        private RawColor4 ConvertColor(System.Drawing.Color color)
        {
            return new RawColor4(color.R / 255f, color.G / 255f, color.B / 255f, color.A / 255f);
        }


    }
}