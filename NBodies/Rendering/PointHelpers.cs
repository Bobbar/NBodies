using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace NBodies.Rendering
{
    public static class PointHelper
    {
        public static PointF Subtract(this PointF pointA, PointF pointB)
        {
            var diffX = pointA.X - pointB.X;
            var diffY = pointA.Y - pointB.Y;

            return new PointF(diffX, diffY);
        }

        public static PointF Subtract(this Point pointA, PointF pointB)
        {
            var diffX = pointA.X - pointB.X;
            var diffY = pointA.Y - pointB.Y;

            return new PointF(diffX, diffY);
        }

        public static PointF Subtract(this Point pointA, Point pointB)
        {
            var diffX = pointA.X - pointB.X;
            var diffY = pointA.Y - pointB.Y;

            return new PointF(diffX, diffY);
        }

        public static PointF Add(this PointF pointA, PointF pointB)
        {
            return new PointF(pointA.X + pointB.X, pointA.Y + pointB.Y);
        }

        public static PointF FromDouble(this PointF d, double X, double Y)
        {
            return new PointF((float)X, (float)Y);
        }

        public static double Distance(this PointF pointA, PointF pointB)
        {
            return Math.Sqrt(Math.Pow(pointA.X - pointB.X, 2) + Math.Pow(pointA.Y - pointB.Y, 2));
        }

        public static bool PointInsideCircle(PointF circleLoc, float circleRadius, PointF testPoint)
        {
            var dist = testPoint.Distance(circleLoc);

            if (dist <= circleRadius)
                return true;

            return false;
        }
    }
}
