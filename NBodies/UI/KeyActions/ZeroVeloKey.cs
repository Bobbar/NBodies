using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Physics;

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

        public override void DoKeyUp()
        {
           // throw new NotImplementedException();
        }

        public override void DoMouseDown(MouseButtons button, PointF mouseLoc)
        {
           // throw new NotImplementedException();
        }

        public override void DoMouseMove(PointF mouseLoc)
        {
           // throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
           // throw new NotImplementedException();
        }

        public override void DoWheelAction(int wheelValue)
        {
            //throw new NotImplementedException();
        }
    }
}
