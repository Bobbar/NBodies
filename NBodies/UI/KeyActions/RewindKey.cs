using NBodies.Rendering;
using System.Drawing;
using System.Windows.Forms;
using NBodies.Physics;

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
                Overlay.Value = $@"Rewind: { BodyManager.StateIdx } / { BodyManager.StateCount }";
            }
            else
            {
                Overlay.Value = $@"Rewind: No frames...";
            }
        }
    }
}