﻿using System;
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

        public static PointF Multi(this PointF pointA, float value)
        {
            return new PointF(pointA.X * value, pointA.Y * value);
        }

        public static float DistanceSqrt(this PointF pointA, PointF pointB)
        {
            return (float)Math.Sqrt(Math.Pow(pointA.X - pointB.X, 2) + Math.Pow(pointA.Y - pointB.Y, 2));
        }

        public static float Distance(this PointF pointA, PointF pointB)
        {
            return (float)(Math.Pow(pointA.X - pointB.X, 2) + Math.Pow(pointA.Y - pointB.Y, 2));
        }

        public static float Length(this PointF pointA)
        {
            return (float)(Math.Pow(pointA.X, 2) + Math.Pow(pointA.Y, 2));
        }

        public static float LengthSqrt(this PointF pointA)
        {
            return (float)Math.Sqrt((Math.Pow(pointA.X, 2) + Math.Pow(pointA.Y, 2)));
        }

        public static bool PointInsideCircle(PointF circleLoc, float circleRadius, PointF testPoint)
        {
            //var dist = testPoint.DistanceSqrt(circleLoc);

            //if (dist <= circleRadius)
            //    return true;

            //return false;


            var dist = Math.Pow((testPoint.X - circleLoc.X), 2) + Math.Pow((testPoint.Y - circleLoc.Y), 2);

            if (dist <= circleRadius * circleRadius)
            {
                return true;
            }

            return false;

        }

        public static bool IsIntersecting(PointF a, PointF b, PointF c, PointF d)
        {
            float denominator = ((b.X - a.X) * (d.Y - c.Y)) - ((b.Y - a.Y) * (d.X - c.X));
            float numerator1 = ((a.Y - c.Y) * (d.X - c.X)) - ((a.X - c.X) * (d.Y - c.Y));
            float numerator2 = ((a.Y - c.Y) * (b.X - a.X)) - ((a.X - c.X) * (b.Y - a.Y));

            if (denominator == 0)
                return numerator1 == 0 && numerator2 == 0;

            float r = numerator1 / denominator;
            float s = numerator2 / denominator;

            return (r >= 0 && r <= 1) && (s >= 0 && s <= 1);
        }

        public static Rectangle ToRectangle(this RectangleF rect)
        {
            return new Rectangle((int)rect.X, (int)rect.Y, (int)rect.Width, (int)rect.Height);
        }
    }
}
