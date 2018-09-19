using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace NBodies.Rendering
{
    public static class RenderVars
    {
        public static float CurrentScale { get; set; }
        public static PointF ScaleOffset { get; set; } = new PointF();

        public static PointF ViewportOffset { get; set; } = new PointF();


    }
}
