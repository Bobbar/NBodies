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
    public class GasKKey : KeyAction
    {
        private Keys _myKey = Keys.G;

        public GasKKey()
        {
            AddKey(_myKey);
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

        public override void DoMouseDown(MouseButtons button, PointF mouseLoc)
        {
            //  throw new NotImplementedException();
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
            if (KeyDownStates[_myKey])
            {
                MainLoop.GasK += (wheelValue * 0.01f);
                Overlay.Value = $@"GasK: {MainLoop.GasK}";
            }
        }
    }
}
