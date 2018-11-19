using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing;

namespace NBodies.UI
{
    public static class InputHandler
    {

        public static bool KeysDown = false;

        private static Dictionary<Keys, KeyAction> _actions = new Dictionary<Keys, KeyAction>();

        public static void AddKeyAction(KeyAction keyaction)
        {
            if (!_actions.ContainsKey(keyaction.Key))
            {
                _actions.Add(keyaction.Key, keyaction);
            }
        }

        public static void KeyDown(Keys key)
        {
            KeysDown = true;

            if (_actions.ContainsKey(key))
            {
                _actions[key].IsDown = true;
                _actions[key].DoKeyDown();
            }
        }

        public static void KeyUp(Keys key)
        {
            if (_actions.ContainsKey(key))
            {
                _actions[key].IsDown = false;
                _actions[key].DoKeyUp();
            }

            bool keysDown = false;

            foreach (var a in _actions)
            {
                if (a.Value.IsDown)
                    keysDown = true;
            }

            KeysDown = keysDown;
        }

        public static void MouseWheel(int delta)
        {
            foreach (var a in _actions.Values)
            {
                if (a.IsDown)
                {
                    if (delta > 0)
                    {
                        a.DoWheelAction(1);
                    }
                    else
                    {
                        a.DoWheelAction(-1);
                    }
                }
            }
        }

        public static void MouseMove(Point mouseLoc)
        {
            foreach (var a in _actions.Values)
            {
                if (a.IsDown)
                {
                    a.DoMouseMove(mouseLoc);
                }
            }
        }


    }
}
