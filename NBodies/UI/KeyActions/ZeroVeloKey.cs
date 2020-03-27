using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Physics;
using OpenTK;


namespace NBodies.UI.KeyActions
{
    class ZeroVeloKey : KeyAction
    {
        public ZeroVeloKey()
        {
            AddKey(Keys.X);
            AddKey(Keys.ShiftKey);

           // Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            if (KeyDownStates[Keys.ShiftKey] && KeyDownStates[Keys.X])
            {
                BodyManager.ZeroVelocities();
            }
        }
    }
}
