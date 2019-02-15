using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;
using NBodies.Extensions;

namespace NBodies.UI.KeyActions
{
    public class FPSKey : KeyAction
    {
        public FPSKey()
        {
            AddKey(Keys.F);
            AddKey(Keys.ShiftKey);

            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            if (KeyDownStates[Keys.ShiftKey] && KeyDownStates[Keys.F])
            {
                Overlay.Value = $@"Burst Frames: {MainLoop.RenderBurstFrames}";
                Overlay.Show();
            }
            else if (!KeyDownStates[Keys.ShiftKey] && KeyDownStates[Keys.F])
            {
                Overlay.Value = $@"FPS Max: {MainLoop.TargetFPS}";
                Overlay.Show();
            }
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoMouseDown(MouseButtons button, PointF mouseLoc)
        {
            // throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
            //throw new NotImplementedException();
        }

        public override void DoMouseMove(PointF mouseLoc)
        {
            //throw new NotImplementedException();
        }

        public override void DoWheelAction(int wheelValue)
        {
            if (KeyDownStates[Keys.F])
            {
                if (KeyDownStates[Keys.ShiftKey])
                {
                    MainLoop.RenderBurstFrames += wheelValue;
                    Overlay.Value = $@"Burst Frames: {MainLoop.RenderBurstFrames}";
                }
                else
                {
                    MainLoop.TargetFPS += wheelValue;
                    Overlay.Value = $@"FPS Max: {MainLoop.TargetFPS}";
                }
            }
        }
    }
}
