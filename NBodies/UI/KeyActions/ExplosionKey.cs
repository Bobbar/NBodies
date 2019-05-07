using NBodies.Helpers;
using NBodies.Physics;
using NBodies.Rendering;
using System.Drawing;
using System.Windows.Forms;

namespace NBodies.UI.KeyActions
{
    internal class ExplosionKey : KeyAction
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
                BodyManager.InsertExplosion(ViewportHelpers.ScreenPointToField(mouseLoc), 2500);
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