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
    public class AlphaKey : KeyAction
    {
        public AlphaKey()
        {
            AddKey(Keys.A);
            AddKey(Keys.ControlKey);
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
          
            //    Overlay.Value = "Alpha: " + RenderBase.BodyAlpha;
            //Overlay.Show();
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
            //throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
           // throw new NotImplementedException();
        }

        public override void DoWheelAction(int wheelValue)
        {
            if (KeyDownStates[Keys.ControlKey] && KeyDownStates[Keys.A])
            {
                RenderBase.BodyAlpha += wheelValue;
                Console.WriteLine(RenderBase.BodyAlpha);

                Overlay.Value = "Alpha: " + RenderBase.BodyAlpha;
            }
        }
    }
}
