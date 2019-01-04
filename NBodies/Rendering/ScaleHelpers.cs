﻿using NBodies.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace NBodies.Rendering
{
    public static class ScaleHelpers
    {
        public static PointF ScreenPointToField(PointF point)
        {
            PointF convertedPos = new PointF((point.X / RenderVars.CurrentScale) - RenderVars.ViewportOffset.X - RenderVars.ScaleOffset.X, (point.Y / RenderVars.CurrentScale) - RenderVars.ViewportOffset.Y - RenderVars.ScaleOffset.Y);

            return convertedPos;
        }

        public static PointF FieldPointToScreen(PointF point)
        {
            PointF convertedPos = point.Add(RenderVars.ViewportOffset.Add(RenderVars.ScaleOffset)).Multi(RenderVars.CurrentScale);

            return convertedPos;
        }

        public static PointF FieldPointToScreenUnscaled(PointF point)
        {
            PointF convertedPos = new PointF((point.X / RenderVars.CurrentScale), (point.Y / RenderVars.CurrentScale));

            return convertedPos;
        }
    }
}
