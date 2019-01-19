using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpDX;
//using System.Drawing;

using d2 = SharpDX.Direct2D1;
using d3d = SharpDX.Direct3D11;
using dxgi = SharpDX.DXGI;
using wic = SharpDX.WIC;
using dw = SharpDX.DirectWrite;
using SharpDX.Mathematics;


using NBodies.Physics;
using NBodies.Extensions;

using System.Windows.Forms;

using System.Drawing;

namespace NBodies.Rendering
{
    public static class D2DRenderer
    {
        //  private static Device d2dDevice = new SharpDX.DXGI.Device()
        private static dxgi.SwapChain1 swapChain;
        private static d2.Bitmap1 d2dTarget;

        static SharpDX.Direct2D1.Factory fact = new SharpDX.Direct2D1.Factory(SharpDX.Direct2D1.FactoryType.SingleThreaded);
        static d2.HwndRenderTargetProperties hwndProperties;
        static d2.RenderTargetProperties rndTargProperties;
        static d2.WindowRenderTarget wndRender;

        public static void Init(Control target)
        {
            rndTargProperties = new d2.RenderTargetProperties(new d2.PixelFormat(dxgi.Format.B8G8R8A8_UNorm, d2.AlphaMode.Premultiplied));
            hwndProperties = new d2.HwndRenderTargetProperties();
            hwndProperties.Hwnd = target.Handle;
            hwndProperties.PixelSize = new Size2(target.ClientSize.Width, target.ClientSize.Height);
            hwndProperties.PresentOptions = d2.PresentOptions.None;

            wndRender = new d2.WindowRenderTarget(fact, rndTargProperties, hwndProperties);
        }

        public static void Test(Body[] bodies)
        {






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

            var finalOffset = RenderVars.ViewportOffset.Add(RenderVars.ScaleOffset);


            for (int i = 0; i < bodies.Length; i++)
            {
                var body = bodies[i];
                var bodyLoc = new System.Drawing.PointF((body.LocX - body.Size * 0.5f + finalOffset.X), (body.LocY - body.Size * 0.5f + finalOffset.Y));

               // d2dContext.FillEllipse(new SharpDX.Direct2D1.Ellipse(new SharpDX.Mathematics.Interop.RawVector2(bodyLoc.X, bodyLoc.Y), body.Size, body.Size), new SharpDX.Direct2D1.SolidColorBrush(d2dContext, new SharpDX.Mathematics.Interop.RawColor4(255, 255, 255, 255)));
                wndRender.FillEllipse(new SharpDX.Direct2D1.Ellipse(new SharpDX.Mathematics.Interop.RawVector2(bodyLoc.X, bodyLoc.Y), body.Size, body.Size), scenebrush);
               
            
           

            }

           // d2dContext.EndDraw();

            //wndRender.Flush();
            wndRender.EndDraw();


        }
    }
}
