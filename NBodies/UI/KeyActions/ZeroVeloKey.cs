using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Physics;
using OpenTK;


namespace NBodies.UI.KeyActions
{
    class ZeroVeloKey : KeyAction
    {
        public ZeroVeloKey(params Keys[] keys) : base(keys) { }

        public override void DoKeyDown()
        {
            BodyManager.ZeroVelocities();
        }
    }
}
