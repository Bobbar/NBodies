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
        private Keys _myKey = Keys.V;

        public ViscosityKey()
        {
            AddKey(_myKey);
            AddKey(Keys.ControlKey);
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }
        public override void DoKeyDown()
        {
            if (KeyDownStates[_myKey] && KeyDownStates[Keys.ControlKey])
            {
                Overlay.Value = "Viscosity: " + MainLoop.Viscosity;
                Overlay.Show();
            }
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
            // throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
            // throw new NotImplementedException();
        }

        public override void DoWheelAction(int wheelValue)
        {
            if (KeyDownStates[_myKey] && KeyDownStates[Keys.ControlKey])
            {
                MainLoop.Viscosity += (wheelValue * 0.1f);
                Overlay.Value = "Viscosity: " + MainLoop.Viscosity;
            }
        }
    }
}
