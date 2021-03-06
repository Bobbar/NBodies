﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;
using System.Drawing;

namespace NBodies.UI.KeyActions
{
    class ViscosityKey : KeyAction
    {
        private Keys _myKey = Keys.V;

        public ViscosityKey()
        {
            AddKey(_myKey);
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }
        public override void DoKeyDown()
        {
            Overlay.Value = "Viscosity: " +  MainLoop.Viscosity;
            Overlay.Show();
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
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
            if (KeyDownStates[_myKey])
            {
                MainLoop.Viscosity += (wheelValue * 0.1f);
                Overlay.Value = "Viscosity: " + MainLoop.Viscosity;
            }
        }
    }
}
