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
        private int _displayCombo = -1;
        private int _scaleCombo = -1;

        public DisplayStyleKey()
        {
            _displayCombo = AddKeyCombo(new KeyCombo(Keys.ControlKey, Keys.ShiftKey, Keys.S));
            _scaleCombo = AddKeyCombo(new KeyCombo(Keys.ControlKey, Keys.S));

            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown(int comboId)
        {
            if (comboId == _displayCombo)
            {
                Overlay.Value = "Display: " + RenderVars.DisplayStyle.ToString();
                Overlay.Show();
            }
            else if (comboId == _scaleCombo)
            {
                Overlay.Value = "Style Scale: " + RenderVars.StyleScaleMax;
                Overlay.Show();
            }
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoWheelAction(int wheelValue, int comboId)
        {
            if (comboId == _displayCombo)
            {
                int max = Enum.GetValues(typeof(DisplayStyle)).Cast<int>().Max();
                int min = Enum.GetValues(typeof(DisplayStyle)).Cast<int>().Min();

                if ((int)RenderVars.DisplayStyle + wheelValue <= max && (int)RenderVars.DisplayStyle + wheelValue >= min)
                {
                    RenderVars.DisplayStyle += wheelValue;
                    Overlay.Value = "Display: " + RenderVars.DisplayStyle.ToString();

                }
            }
            else if (comboId == _scaleCombo)
            {
                RenderVars.StyleScaleMax += wheelValue * 2;
                Overlay.Value = "Style Scale: " + RenderVars.StyleScaleMax;
            }
        }
    }
}

