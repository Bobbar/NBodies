using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NBodies.UI.KeyActions
{
    public class SimpleKey : KeyAction
    {
        public SimpleKey(Keys key) : base(key)
        {
        }

        public override void DoKeyDown()
        {
            //throw new NotImplementedException();
        }

        public override void DoKeyUp()
        {
           // throw new NotImplementedException();
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
            //throw new NotImplementedException();
        }
    }
}
