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
    public class GasKKey : KeyAction
    {
        public GasKKey(Keys key) : base(key)
        {
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            Overlay.Value = $@"GasK: {MainLoop.GasK}";
            Overlay.Show();
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoWheelAction(int wheelValue)
        {
            MainLoop.GasK += (wheelValue * 0.01f);
            Overlay.Value = $@"GasK: {MainLoop.GasK}";
        }
    }
}
