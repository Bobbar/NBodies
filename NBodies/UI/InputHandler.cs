using NBodies.Rendering;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using OpenTK;

namespace NBodies.UI
{
    public static class InputHandler
    {
        public static bool KeysDown = false;
        public static bool MouseIsDown = false;

        private static List<KeyAction> _actions = new List<KeyAction>();

        public static Dictionary<Keys, bool> KeyDownStates = new Dictionary<Keys, bool>();


        public static void AddKeyAction(KeyAction keyaction)
        {
            _actions.Add(keyaction);

            if (keyaction.Overlay != null)
            {
                RenderVars.AddOverlay(keyaction.Overlay);
            }
        }

        public static void KeyDown(Keys key)
        {
            KeysDown = true;

            if (!KeyDownStates.ContainsKey(key))
                KeyDownStates.Add(key, true);

            if (!KeyDownStates[key])
                KeyDownStates[key] = true;


            foreach (var action in _actions)
            {
                if (action.KeyDownStates.ContainsKey(key))
                {
                    if (!action.KeyDownStates[key])
                    {
                        action.KeyDownStates[key] = true;
                        action.KeyDown();
                    }
                }
            }
        }

        public static void KeyUp(Keys key)
        {
            bool keysDown = false;

            if (!KeyDownStates.ContainsKey(key))
                KeyDownStates.Add(key, false);

            if (KeyDownStates[key])
                KeyDownStates[key] = false;

            foreach (var state in KeyDownStates.Values)
            {
                if (state)
                    keysDown = true;
            }


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

        public static void MouseDown(MouseButtons buttons, Vector3 loc)
        {
            MouseIsDown = true;

            foreach (var action in _actions)
            {
                action.MouseDown(buttons, loc);
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
            if (KeyDownStates.ContainsKey(key))
            {
                return KeyDownStates[key];
            }

            return false;

            //foreach (var action in _actions)
            //{
            //    if (action.KeyDownStates.ContainsKey(key))
            //    {
            //        return action.KeyDownStates[key];
            //    }
            //}

           // return false;
        }
    }
}