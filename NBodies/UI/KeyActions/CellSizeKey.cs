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
    public class CellSizeKey : KeyAction
    {
        public CellSizeKey()
        {
            AddKey(Keys.C);
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }
        public override void DoKeyDown()
        {
            Overlay.Value = "Cell Size: " + Math.Pow(2, MainLoop.CellSizeExp).ToString();
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
           // throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
           // throw new NotImplementedException();
        }

        public override void DoWheelAction(int wheelValue)
        {
            if (KeyDownStates[Keys.C])
            {
                MainLoop.CellSizeExp += wheelValue;
                Overlay.Value = "Cell Size: " + Math.Pow(2, MainLoop.CellSizeExp).ToString();
            }
        }
    }
}
