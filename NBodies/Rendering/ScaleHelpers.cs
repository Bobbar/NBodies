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
        public static PointF ScaleMousePosRelative(PointF mousePos)
        {
            PointF convertedPos = new PointF((mousePos.X / RenderVars.CurrentScale) - RenderVars.ViewportOffset.X - RenderVars.ScaleOffset.X, (mousePos.Y / RenderVars.CurrentScale) - RenderVars.ViewportOffset.Y - RenderVars.ScaleOffset.Y);

            return convertedPos;
        }

        public static PointF ScaleMousePosExact(PointF mousePos)
        {
            PointF convertedPos = new PointF((mousePos.X / RenderVars.CurrentScale), (mousePos.Y / RenderVars.CurrentScale));

            return convertedPos;
        }
    }
}
