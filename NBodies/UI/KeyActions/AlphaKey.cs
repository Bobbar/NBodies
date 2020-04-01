﻿using System;
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
    public class AlphaKey : KeyAction
    {
        public AlphaKey(params Keys[] keys) : base(keys)
        {
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            Overlay.Value = "Alpha: " + RenderVars.BodyAlpha;
            Overlay.Show();
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoWheelAction(int wheelValue)
        {
            RenderVars.BodyAlpha += wheelValue;
            Overlay.Value = "Alpha: " + RenderVars.BodyAlpha;
        }
    }
}
