using NBodies.Rendering;
using System.Drawing;
using System.Windows.Forms;
using NBodies.Physics;
using OpenTK;
using System;

namespace NBodies.UI.KeyActions
{
    public class RewindKey : KeyAction
    {
        private Keys myKey = Keys.R;

        public RewindKey()
        {
            AddKey(myKey);
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            MainLoop.WaitForPause();

            RefreshOverlay();
            Overlay.Show();
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();

            MainLoop.ResumePhysics();
        }

        public override void DoMouseDown(MouseButtons button, Vector3 loc)
        {
            //throw new NotImplementedException();
        }

        public override void DoMouseDown(MouseButtons button, PointF mouseLoc)
        {
            // throw new NotImplementedException();
        }

        public override void DoMouseMove(PointF mouseLoc)
        {
            //  throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
            // throw new NotImplementedException();
        }

        public override void DoWheelAction(int wheelValue)
        {
            if (KeyDownStates[myKey])
            {
                if (wheelValue > 0)
                {
                    BodyManager.FastForwardState();
                }
                else
                {
                    BodyManager.RewindState();
                }

                RefreshOverlay();
                Overlay.Show();
            }
        }

        internal void RefreshOverlay()
        {
            if (BodyManager.StateCount > 0)
            {
                //Overlay.Value = $@"Rewind: { BodyManager.StateIdx } / { BodyManager.StateCount }";

                string pbar = PBar(BodyManager.StateIdx, 20, BodyManager.StateCount);
                Overlay.Value = $@"Rewind: { pbar }";
            }
            else
            {
                Overlay.Value = $@"Rewind: No frames...";
            }
        }

        internal string PBar(int value, int width, int maxValue)
        {
            // Clamp.
            value = Math.Min(value, maxValue);

            // Styling.
            const char cursor = 'V';
            const char line = '_';
            const char endCap = '|';

            // Compute position.
            float step = width / (float)maxValue;
            int cursorPos = (int)(value * step);

            char[] bar = new char[width];

            // Build the bar.
            for (int i = 0; i < width; i++)
            {
                if (cursorPos == i)
                    bar[i] = cursor;
                else
                    bar[i] = line;
            }

            // Handle last condition.
            if (cursorPos == width)
                bar[width - 1] = cursor;

            // Add caps.
            string completeBar = $"{endCap}{new string(bar)}{endCap}";

            return completeBar;
        }
    }
}