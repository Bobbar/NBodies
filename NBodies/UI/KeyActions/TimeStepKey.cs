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
    public class TimeStepKey : KeyAction
    {
        public TimeStepKey()
        {
            AddKey(Keys.T);
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            Overlay.Value = "Timestep: " + Math.Round(MainLoop.TimeStep, 5);
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
            //  throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
            // throw new NotImplementedException();
        }

        public override void DoWheelAction(int wheelValue)
        {
            if (KeyDownStates[Keys.T])
            {
                MainLoop.TimeStep += 0.0001f * wheelValue;
                Overlay.Value = "Timestep: " + Math.Round(MainLoop.TimeStep, 5);
            }
        }
    }
}
