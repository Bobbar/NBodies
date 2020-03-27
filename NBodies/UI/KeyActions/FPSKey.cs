using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;
using NBodies.Extensions;
using OpenTK;

namespace NBodies.UI.KeyActions
{
    public class FPSKey : KeyAction
    {
        public FPSKey()
        {
            AddKey(Keys.F);

            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            if (KeyDownStates[Keys.F])
            {
                Overlay.Value = $@"FPS Max: {MainLoop.TargetFPS}";
                Overlay.Show();
            }
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoWheelAction(int wheelValue)
        {
            if (KeyDownStates[Keys.F])
            {
                MainLoop.TargetFPS += wheelValue;
                Overlay.Value = $@"FPS Max: {MainLoop.TargetFPS}";
            }
        }
    }
}
