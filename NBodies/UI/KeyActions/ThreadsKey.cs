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
    public class ThreadsKey : KeyAction
    {
        private Keys myKey = Keys.B;

        public ThreadsKey()
        {
            AddKey(myKey);
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            Overlay.Value = "Threads Per Block: " + (int)Math.Pow(2, MainLoop.ThreadsPerBlock);
            Overlay.Show();
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoMouseDown(MouseButtons button, PointF mouseLoc)
        {
            //throw new NotImplementedException();
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
            if (KeyDownStates[myKey])
            {
                MainLoop.ThreadsPerBlock += wheelValue;
                Overlay.Value = "Threads Per Block: " + (int)Math.Pow(2,MainLoop.ThreadsPerBlock);
            }
        }
    }
}
