using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;
using NBodies.Extensions;
using OpenTK;

namespace NBodies.UI.KeyActions
{

    public class PointSpriteTexKey : KeyAction
    {
        public PointSpriteTexKey()
        {
            AddKey(Keys.H);
        }

        public override void DoKeyDown()
        {
        }

        public override void DoKeyUp()
        {
        }

        public override void DoMouseDown(MouseButtons button, Vector3 loc)
        {
            //throw new NotImplementedException();
        }

        public override void DoMouseDown(MouseButtons button, PointF mouseLoc)
        {
            // throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
            //throw new NotImplementedException();
        }

        public override void DoMouseMove(PointF mouseLoc)
        {
            //throw new NotImplementedException();
        }

        public override void DoWheelAction(int wheelValue)
        {
            if (KeyDownStates[Keys.H])
            {
                RenderVars.PointSpriteTexIdx += wheelValue;
            }
        }
    }
}


