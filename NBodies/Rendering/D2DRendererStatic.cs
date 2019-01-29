using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using SharpDX;
//using System.Drawing;

using d2 = SharpDX.Direct2D1;
using d3d = SharpDX.Direct3D11;
using dxgi = SharpDX.DXGI;
using wic = SharpDX.WIC;
using dw = SharpDX.DirectWrite;
using SharpDX.Mathematics;

using SharpDX.Mathematics.Interop;

using NBodies.Physics;
using NBodies.Extensions;

using System.Windows.Forms;

using System.Drawing;

namespace NBodies.Rendering
{
    public static class D2DRendererStatic
    {
        private static Control _target;
        //  private static Device d2dDevice = new SharpDX.DXGI.Device()
        private static dxgi.SwapChain1 swapChain;
        private static d2.Bitmap1 d2dTarget;

        static SharpDX.Direct2D1.Factory fact = new SharpDX.Direct2D1.Factory(SharpDX.Direct2D1.FactoryType.SingleThreaded);
        static d2.HwndRenderTargetProperties hwndProperties;
        static d2.RenderTargetProperties rndTargProperties;
        static d2.WindowRenderTarget wndRender;

        private static Matrix3x2 _transform = new Matrix3x2();

        private static Size _prevSize;
        private static float _prevScale;

        private static System.Drawing.RectangleF _cullTangle;


        private static System.Diagnostics.Stopwatch timer = new System.Diagnostics.Stopwatch();

        public static void Init(Control target)
        {
            _target = target;
            rndTargProperties = new d2.RenderTargetProperties(new d2.PixelFormat(dxgi.Format.B8G8R8A8_UNorm, d2.AlphaMode.Premultiplied));
            //  rndTargProperties = new d2.RenderTargetProperties(new d2.PixelFormat(dxgi.Format., d2.AlphaMode.Premultiplied));

            hwndProperties = new d2.HwndRenderTargetProperties();
            hwndProperties.Hwnd = target.Handle;
            hwndProperties.PixelSize = new Size2(target.ClientSize.Width, target.ClientSize.Height);
            hwndProperties.PresentOptions = d2.PresentOptions.Immediately;
            rndTargProperties.MinLevel = d2.FeatureLevel.Level_10;

            wndRender = new d2.WindowRenderTarget(fact, rndTargProperties, hwndProperties);

            _prevSize = target.ClientSize;
        }

        public static void CheckScale()
        {
            if (_target.ClientSize != _prevSize)
            {
                // Init(_target);

                //  hwndProperties.PixelSize = new Size2(_target.ClientSize.Width, _target.ClientSize.Height);
                UpdateGraphicsScale();
            }

            if (_prevScale != RenderVars.CurrentScale)
            {
                UpdateGraphicsScale();
            }
        }

        private static void UpdateGraphicsScale()
        {

            _transform.ScaleVector = new Vector2(RenderVars.CurrentScale, RenderVars.CurrentScale);
            wndRender.Transform = _transform;
            _prevScale = RenderVars.CurrentScale;
        }

        public static void DrawBodies(Body[] bodies)
        {
            timer.Restart();


            CheckScale();

            var finalOffset = RenderVars.ViewportOffset.Add(RenderVars.ScaleOffset);


            _cullTangle = new System.Drawing.RectangleF(0 - finalOffset.X, 0 - finalOffset.Y, _target.ClientSize.Width / RenderVars.CurrentScale, _target.ClientSize.Height / RenderVars.CurrentScale);


            //        var defaultDevice = new SharpDX.Direct3D11.Device(SharpDX.Direct3D.DriverType.Hardware,
            //                                                         d3d.DeviceCreationFlags.VideoSupport
            //                                                         | d3d.DeviceCreationFlags.BgraSupport
            //                                                         | d3d.DeviceCreationFlags.None);

            //        var d3dDevice = defaultDevice.QueryInterface<d3d.Device1>();
            //        var dxgiDevice = d3dDevice.QueryInterface<dxgi.Device>();
            //        var d2dDevice = new d2.Device(dxgiDevice);
            //        var d2dContext = new d2.DeviceContext(d2dDevice, d2.DeviceContextOptions.None);

            //        d2.BitmapProperties1 properties = new d2.BitmapProperties1(new d2.PixelFormat(SharpDX.DXGI.Format.B8G8R8A8_UNorm, SharpDX.Direct2D1.AlphaMode.Premultiplied),
            //96, 96, d2.BitmapOptions.Target | d2.BitmapOptions.CannotDraw);

            //        dxgi.Surface backBuffer = swapChain.GetBackBuffer<dxgi.Surface>(0);

            //        d2dTarget = new d2.Bitmap1(d2dContext, backBuffer, properties);

            //d2dContext.Target = d2dTarget;
            //d2dContext.BeginDraw();

            //d2dContext.Clear(new SharpDX.Mathematics.Interop.RawColor4(0, 0, 0, 255));



            wndRender.BeginDraw();

            wndRender.Clear(new SharpDX.Mathematics.Interop.RawColor4(0, 0, 0, 255));

            //var scenebrush = new d2.SolidColorBrush(wndRender, new SharpDX.Mathematics.Interop.RawColor4(66, 136, 244, 100));
            var scenebrush = new d2.SolidColorBrush(wndRender, new SharpDX.Mathematics.Interop.RawColor4(0, 0, 1, 0.75f));


            var drawEllip = new SharpDX.Direct2D1.Ellipse(new SharpDX.Mathematics.Interop.RawVector2(), 0, 0);


            for (int i = 0; i < bodies.Length; i++)
            {
                var body = bodies[i];
                //var bodyLoc = new System.Drawing.PointF((body.LocX - body.Size * 0.5f + finalOffset.X), (body.LocY - body.Size * 0.5f + finalOffset.Y));
                var bodyLoc = new System.Drawing.PointF((body.LocX + finalOffset.X), (body.LocY + finalOffset.Y));


                if (!_cullTangle.Contains(body.LocX, body.LocY)) continue;

                drawEllip.Point.X = bodyLoc.X;
                drawEllip.Point.Y = bodyLoc.Y;
                drawEllip.RadiusX = body.Size * 0.5f;
                drawEllip.RadiusY = body.Size * 0.5f;

                var bc = System.Drawing.Color.FromArgb(body.Color);

                scenebrush.Color = new SharpDX.Mathematics.Interop.RawColor4(bc.R / 255f, bc.G / 255f, bc.B / 255f, 0.75f);

                // d2dContext.FillEllipse(new SharpDX.Direct2D1.Ellipse(new SharpDX.Mathematics.Interop.RawVector2(bodyLoc.X, bodyLoc.Y), body.Size, body.Size), new SharpDX.Direct2D1.SolidColorBrush(d2dContext, new SharpDX.Mathematics.Interop.RawColor4(255, 255, 255, 255)));
                //wndRender.FillEllipse(new SharpDX.Direct2D1.Ellipse(new SharpDX.Mathematics.Interop.RawVector2(bodyLoc.X, bodyLoc.Y), body.Size / 2, body.Size / 2), scenebrush);
                //wndRender.FillEllipse(drawEllip, scenebrush);

                wndRender.FillEllipse(drawEllip, scenebrush);





            }

            //   var blah = new Matrix3x2();


            // wndRender.Transform = new RawMatrix3x2()

            // d2dContext.EndDraw();

            //wndRender.Flush();
            wndRender.EndDraw();


            Console.WriteLine(timer.ElapsedMilliseconds);


        }



        public async static Task DrawBodiesAsync(Body[] bodies, ManualResetEventSlim completeCallback)
        {

            completeCallback.Reset();

            await Task.Run(() =>
            {

                timer.Restart();


                CheckScale();

                var finalOffset = RenderVars.ViewportOffset.Add(RenderVars.ScaleOffset);


                _cullTangle = new System.Drawing.RectangleF(0 - finalOffset.X, 0 - finalOffset.Y, _target.ClientSize.Width / RenderVars.CurrentScale, _target.ClientSize.Height / RenderVars.CurrentScale);


                //        var defaultDevice = new SharpDX.Direct3D11.Device(SharpDX.Direct3D.DriverType.Hardware,
                //                                                         d3d.DeviceCreationFlags.VideoSupport
                //                                                         | d3d.DeviceCreationFlags.BgraSupport
                //                                                         | d3d.DeviceCreationFlags.None);

                //        var d3dDevice = defaultDevice.QueryInterface<d3d.Device1>();
                //        var dxgiDevice = d3dDevice.QueryInterface<dxgi.Device>();
                //        var d2dDevice = new d2.Device(dxgiDevice);
                //        var d2dContext = new d2.DeviceContext(d2dDevice, d2.DeviceContextOptions.None);

                //        d2.BitmapProperties1 properties = new d2.BitmapProperties1(new d2.PixelFormat(SharpDX.DXGI.Format.B8G8R8A8_UNorm, SharpDX.Direct2D1.AlphaMode.Premultiplied),
                //96, 96, d2.BitmapOptions.Target | d2.BitmapOptions.CannotDraw);

                //        dxgi.Surface backBuffer = swapChain.GetBackBuffer<dxgi.Surface>(0);

                //        d2dTarget = new d2.Bitmap1(d2dContext, backBuffer, properties);

                //d2dContext.Target = d2dTarget;
                //d2dContext.BeginDraw();

                //d2dContext.Clear(new SharpDX.Mathematics.Interop.RawColor4(0, 0, 0, 255));



                wndRender.BeginDraw();

                wndRender.Clear(new SharpDX.Mathematics.Interop.RawColor4(0, 0, 0, 255));

                //var scenebrush = new d2.SolidColorBrush(wndRender, new SharpDX.Mathematics.Interop.RawColor4(66, 136, 244, 100));
                var scenebrush = new d2.SolidColorBrush(wndRender, new SharpDX.Mathematics.Interop.RawColor4(0, 0, 1, 0.75f));


                var drawEllip = new SharpDX.Direct2D1.Ellipse(new SharpDX.Mathematics.Interop.RawVector2(), 0, 0);


                for (int i = 0; i < bodies.Length; i++)
                {
                    var body = bodies[i];
                    //var bodyLoc = new System.Drawing.PointF((body.LocX - body.Size * 0.5f + finalOffset.X), (body.LocY - body.Size * 0.5f + finalOffset.Y));
                    var bodyLoc = new System.Drawing.PointF((body.LocX + finalOffset.X), (body.LocY + finalOffset.Y));


                    if (!_cullTangle.Contains(body.LocX, body.LocY)) continue;

                    drawEllip.Point.X = bodyLoc.X;
                    drawEllip.Point.Y = bodyLoc.Y;
                    drawEllip.RadiusX = body.Size * 0.5f;
                    drawEllip.RadiusY = body.Size * 0.5f;

                    var bc = System.Drawing.Color.FromArgb(body.Color);

                    scenebrush.Color = new SharpDX.Mathematics.Interop.RawColor4(bc.R / 255f, bc.G / 255f, bc.B / 255f, 0.75f);

                    // d2dContext.FillEllipse(new SharpDX.Direct2D1.Ellipse(new SharpDX.Mathematics.Interop.RawVector2(bodyLoc.X, bodyLoc.Y), body.Size, body.Size), new SharpDX.Direct2D1.SolidColorBrush(d2dContext, new SharpDX.Mathematics.Interop.RawColor4(255, 255, 255, 255)));
                    //wndRender.FillEllipse(new SharpDX.Direct2D1.Ellipse(new SharpDX.Mathematics.Interop.RawVector2(bodyLoc.X, bodyLoc.Y), body.Size / 2, body.Size / 2), scenebrush);
                    //wndRender.FillEllipse(drawEllip, scenebrush);

                    wndRender.FillEllipse(drawEllip, scenebrush);




                }

                //   var blah = new Matrix3x2();


                // wndRender.Transform = new RawMatrix3x2()

                // d2dContext.EndDraw();

                //wndRender.Flush();
                wndRender.EndDraw();

            });

            Console.WriteLine(timer.ElapsedMilliseconds);

            completeCallback.Set();
        }


    }
}
