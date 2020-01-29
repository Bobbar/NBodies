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
        Keys _myKey = Keys.K;

        public KernelSizeKey()
        {
            AddKey(_myKey);
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

        public override void DoMouseDown(MouseButtons button, Vector3 loc)
        {
            //throw new NotImplementedException();
        }

        public override void DoMouseDown(MouseButtons button, PointF mouseLoc)
        {
           // throw new NotImplementedException();
        }

        public override void DoMouseMove(PointF mouseLoc)
        {
            //throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
          //  throw new NotImplementedException();
        }

        public override void DoWheelAction(int wheelValue)
        {
            if (KeyDownStates[_myKey])
            {
                MainLoop.KernelSize += wheelValue * 0.001f;
                Overlay.Value = "Kernel Size: " + Math.Round(MainLoop.KernelSize,2);
            }
        }
    }
}
