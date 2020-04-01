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


            var currentCombo = GetCurrentCombo();

            foreach (var action in _actions)
            {
                foreach (var combo in action.KeyCombos)
                {
                    if (combo.Value.Equals(currentCombo))
                    {
                        action.KeyDown(combo.Key);
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
                foreach (var combo in action.KeyCombos)
                    if (combo.Value.Contains(key))
                        action.KeyUp();
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

            var currentCombo = GetCurrentCombo();

            foreach (var action in _actions)
            {
                foreach (var combo in action.KeyCombos)
                {
                    if (combo.Value.Equals(currentCombo))
                        action.MouseDown(buttons, loc);
                }
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
            var currentCombo = GetCurrentCombo();

            foreach (var action in _actions)
            {
                foreach (var combo in action.KeyCombos)
                {
                    if (combo.Value.Equals(currentCombo))
                    {
                        if (delta > 0)
                        {
                            action.MouseWheel(1);
                            action.MouseWheel(1, combo.Key);
                        }
                        else
                        {
                            action.MouseWheel(-1);
                            action.MouseWheel(-1, combo.Key);
                        }
                    }
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
        }

        private static KeyCombo GetCurrentCombo()
        {
            var currentCombo = new KeyCombo();
            foreach (var k in KeyDownStates)
            {
                if (k.Value == true)
                    currentCombo.AddKey(k.Key);
            }

            return currentCombo;
        }
    }
}