using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;
using OpenTK;

namespace NBodies.UI.KeyActions
{
    public class DisplayStyleKey : KeyAction
    {
        public DisplayStyleKey()
        {
            AddKey(Keys.S);
            AddKey(Keys.ShiftKey);
            AddKey(Keys.ControlKey);

            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }
        public override void DoKeyDown()
        {
            if (KeyDownStates[Keys.ControlKey] && KeyDownStates[Keys.ShiftKey] && KeyDownStates[Keys.S])
            {
                Overlay.Value = "Display: " + RenderBase.DisplayStyle.ToString();
                Overlay.Show();
            }
            else if (!KeyDownStates[Keys.ControlKey] && KeyDownStates[Keys.S])
            {
                Overlay.Value = "Style Scale: " + RenderBase.StyleScaleMax;
                Overlay.Show();
            }
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoMouseDown(MouseButtons button, Vector3 loc)
        {
            //throw new NotImplementedException();
        }

        public override void DoMouseDown(MouseButtons button, PointF mouseLoc)
        {
            //throw new NotImplementedException();
        }

        public override void DoMouseMove(PointF mouseLoc)
        {
            //throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
            // throw new NotImplementedException();
        }

        public override void DoWheelAction(int wheelValue)
        {
            if (KeyDownStates[Keys.ControlKey] && KeyDownStates[Keys.S])
            {
                if (KeyDownStates[Keys.ShiftKey])
                {
                    int max = Enum.GetValues(typeof(DisplayStyle)).Cast<int>().Max();
                    int min = Enum.GetValues(typeof(DisplayStyle)).Cast<int>().Min();

                    if ((int)RenderBase.DisplayStyle + wheelValue <= max && (int)RenderBase.DisplayStyle + wheelValue >= min)
                    {
                        RenderBase.DisplayStyle += wheelValue;
                        Overlay.Value = "Display: " + RenderBase.DisplayStyle.ToString();

                    }
                }
                else
                {
                    RenderBase.StyleScaleMax += wheelValue * 2;
                    Overlay.Value = "Style Scale: " + RenderBase.StyleScaleMax;
                }
            }
        }
    }
}
