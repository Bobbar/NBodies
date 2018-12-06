using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace NBodies.Shapes
{
    public struct ColoredLine
    {
        public Color Color;
        public PointF PointA;
        public PointF PointB;

        public ColoredLine(Color color, PointF pointA, PointF pointB)
        {
            Color = color;
            PointA = pointA;
            PointB = pointB;
        }
    }
}
