using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing;

namespace NBodies.UI
{
    public class KeyAction
    {
        public readonly Keys Key;
        public Action<int> WheelAction;
        public Action KeyDownAction;
        public Action KeyUpAction;
        public Action<Point> MouseMoveAction;
        public Action<Point> MouseDownAction;

        public bool IsDown = false;

        public KeyAction(Keys key)
        {
            Key = key;
        }

        public KeyAction(Keys key, Action<int> wheelAction)
        {
            Key = key;
            WheelAction = wheelAction;
        }

        public void DoWheelAction(int wheelValue)
        {
            WheelAction?.Invoke(wheelValue);
        }

        public void DoKeyDown()
        {
            KeyDownAction?.Invoke();
        }

        public void DoKeyUp()
        {
            KeyUpAction?.Invoke();
        }

        public void DoMouseMove(Point mouseLoc)
        {
            MouseMoveAction?.Invoke(mouseLoc);
        }

        public void DoMouseDown(Point mouseLoc)
        {
            MouseDownAction?.Invoke(mouseLoc);
        }

    }
}
