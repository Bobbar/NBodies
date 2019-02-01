using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;

namespace NBodies.UI.KeyActions
{
    class ExplosionKey : KeyAction
    {
        public ExplosionKey()
        {
            AddKey(Keys.E);
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            Overlay.Value = "Boom!";
            Overlay.Show();
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoMouseDown(MouseButtons button, PointF mouseLoc)
        {
            if (KeyDownStates[Keys.E])
                BodyManager.InsertExplosion(ScaleHelpers.ScreenPointToField(mouseLoc), 2500);
        }

        public override void DoMouseMove(PointF mouseLoc)
        {
            //throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
            // throw new NotImplementedException();
        }

        public override void DoWheelAction(int wheelValue)
        {
            // throw new NotImplementedException();
        }
    }
}
