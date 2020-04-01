using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;
using System.Drawing;
using OpenTK;

namespace NBodies.UI.KeyActions
{
    class ViscosityKey : KeyAction
    {
        public ViscosityKey(params Keys[] keys) : base(keys)
        {
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            Overlay.Value = "Viscosity: " + MainLoop.Viscosity;
            Overlay.Show();
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoWheelAction(int wheelValue)
        {
            MainLoop.Viscosity += (wheelValue * 0.1f);
            Overlay.Value = "Viscosity: " + MainLoop.Viscosity;
        }
    }
}
