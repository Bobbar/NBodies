using NBodies.Rendering;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;

namespace NBodies.UI
{
    public static class InputHandler
    {
        public static bool KeysDown = false;
        public static bool MouseIsDown = false;

        private static List<KeyAction> _actions = new List<KeyAction>();

        public static void AddKeyAction(KeyAction keyaction)
        {
            _actions.Add(keyaction);

            if (keyaction.Overlay != null)
            {
                RenderBase.AddOverlay(keyaction.Overlay);
            }
        }

        public static void KeyDown(Keys key)
        {
            KeysDown = true;

            foreach (var action in _actions)
            {
                if (action.KeyDownStates.ContainsKey(key))
                {
                    action.KeyDownStates[key] = true;
                    action.KeyDown();
                }
            }
        }

        public static void KeyUp(Keys key)
        {
            bool keysDown = false;

            foreach (var action in _actions)
            {
                if (action.KeyDownStates.ContainsKey(key))
                {
                    action.KeyDownStates[key] = false;
                    action.KeyUp();
                }

                // Check if any keys are down.
                foreach (var state in action.KeyDownStates.Values)
                {
                    if (state == true)
                    {
                        keysDown = true;
                    }
                }
            }

            KeysDown = keysDown;
        }

        public static void MouseDown(MouseButtons buttons, PointF mouseLoc)
        {
            MouseIsDown = true;

            foreach (var action in _actions)
            {
                action.MouseDown(buttons, mouseLoc);
            }
        }

        public static void MouseUp(MouseButtons buttons, PointF mouseLoc)
        {
            MouseIsDown = false;

            foreach (var action in _actions)
            {
                action.MouseUp(buttons, mouseLoc);
            }
        }

        public static void MouseWheel(int delta)
        {
            foreach (var action in _actions)
            {
                if (delta > 0)
                {
                    action.MouseWheel(1);
                }
                else
                {
                    action.MouseWheel(-1);
                }
            }
        }

        public static void MouseMove(Point mouseLoc)
        {
            foreach (var action in _actions)
            {
                action.MouseMove(mouseLoc);
            }
        }

        public static bool KeyIsDown(Keys key)
        {
            foreach (var action in _actions)
            {
                if (action.KeyDownStates.ContainsKey(key))
                {
                    return action.KeyDownStates[key];
                }
            }

            return false;
        }
    }
}