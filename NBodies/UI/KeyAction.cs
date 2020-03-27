﻿using System;
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
        public Dictionary<Keys, bool> KeyDownStates = new Dictionary<Keys, bool>();

        protected PointF _overlayOffset = new PointF(10, 20);

        public KeyAction()
        {
        }

        public KeyAction(Keys key)
        {
            AddKey(key);
        }

        public virtual void DoWheelAction(int wheelValue) { }
        public virtual void DoKeyDown() { }
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

        protected void AddKey(Keys key)
        {
            KeyDownStates.Add(key, false);
        }

        internal void SetOverlayLoc(PointF loc)
        {
            if (Overlay != null)
            {
                Overlay.Location = loc.Subtract(_overlayOffset);
            }
        }


    }
}
