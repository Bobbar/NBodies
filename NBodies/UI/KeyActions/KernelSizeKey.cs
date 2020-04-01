using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;
using OpenTK;

namespace NBodies.UI.KeyActions
{
    public class KernelSizeKey : KeyAction
    {
        public KernelSizeKey(Keys key) : base(key)
        {
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            Overlay.Value = "Kernel Size: " + MainLoop.KernelSize;
            Overlay.Show();
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoWheelAction(int wheelValue)
        {
            MainLoop.KernelSize += wheelValue * 0.001f;
            Overlay.Value = "Kernel Size: " + Math.Round(MainLoop.KernelSize, 2);
        }
    }
}
