using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing;
using NBodies.Rendering;
using NBodies.Extensions;
using OpenTK;

namespace NBodies.UI
{
    public abstract class KeyAction
    {
        public OverlayGraphic Overlay;

        public readonly Dictionary<int, KeyCombo> KeyCombos = new Dictionary<int, KeyCombo>();
        protected PointF _overlayOffset = new PointF(10, 20);

        public KeyAction()
        {
        }

        public KeyAction(Keys key)
        {
            AddKeyCombo(new KeyCombo(key));
        }

        public KeyAction(params Keys[] keys)
        {
            AddKeyCombo(new KeyCombo(keys));
        }

        public virtual void DoWheelAction(int wheelValue) { }
        public virtual void DoWheelAction(int wheelValue, int comboId) { }

        public virtual void DoKeyDown() { }
        public virtual void DoKeyDown(int comboId) { }
        public virtual void DoKeyUp() { }
        public virtual void DoMouseMove(PointF mouseLoc) { }
        public virtual void DoMouseDown(MouseButtons button, PointF mouseLoc) { }
        public virtual void DoMouseDown(MouseButtons button, Vector3 loc) { }
        public virtual void DoMouseUp(MouseButtons button, PointF mouseLoc) { }

        public void MouseDown(MouseButtons button, PointF mouseLoc)
        {
            DoMouseDown(button, mouseLoc);
        }

        public void MouseDown(MouseButtons button, Vector3 loc)
        {
            DoMouseDown(button, loc);
        }

        public void MouseUp(MouseButtons button, PointF mouseLoc)
        {
            DoMouseUp(button, mouseLoc);
        }

        public void KeyDown()
        {
            DoKeyDown();
        }

        public void KeyDown(int comboId)
        {
            DoKeyDown(comboId);
        }

        public void KeyUp()
        {
            DoKeyUp();
        }

        public void MouseMove(PointF mouseLoc)
        {
            SetOverlayLoc(mouseLoc);

            DoMouseMove(mouseLoc);
        }

        public void MouseWheel(int wheelValue)
        {
            DoWheelAction(wheelValue);
        }

        public void MouseWheel(int wheelValue, int comboId)
        {
            DoWheelAction(wheelValue, comboId);
        }

        internal void SetOverlayLoc(PointF loc)
        {
            if (Overlay != null)
            {
                Overlay.Location = loc.Subtract(_overlayOffset);
            }
        }

        protected int AddKeyCombo(KeyCombo combo, int id = -1)
        {
            if (id != -1)
            {
                if (!KeyCombos.ContainsKey(id))
                {
                    KeyCombos.Add(id, combo);
                    return id;
                }
                else
                    throw new Exception($"A key with ID '{id}' already exists.");
            }
            else
            {
                return AddKeyCombo(combo);
            }
        }

        protected int AddKeyCombo(KeyCombo combo)
        {
            int newid = KeyCombos.Count;
            KeyCombos.Add(newid, combo);
            return newid;
        }

    }
}
