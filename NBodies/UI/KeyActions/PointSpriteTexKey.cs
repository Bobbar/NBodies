using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;
using NBodies.Extensions;
using OpenTK;

namespace NBodies.UI.KeyActions
{

    public class PointSpriteTexKey : KeyAction
    {
        public PointSpriteTexKey(Keys key) : base(key)
        {
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            SetText();
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoWheelAction(int wheelValue)
        {
            RenderVars.PointSpriteTexIdx += wheelValue;

            SetText();
        }

        private void SetText()
        {
            string text;

            if (RenderVars.PointSpriteTexIdx <= 0)
                text = "Sprite: Shader Spheres";
            else
                text = $"Sprite: {RenderVars.PointSpriteTexIdx}";

            Overlay.Value = text;
            Overlay.Show();
        }
    }
}


