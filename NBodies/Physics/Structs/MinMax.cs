﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public class MinMax
    {
        public long MinX;
        public long MinY;
        public long MaxX;
        public long MaxY;

        public MinMax()
        {
            MinX = int.MaxValue;
            MinY = int.MaxValue;
            MaxX = int.MinValue;
            MaxY = int.MinValue;
        }

        public MinMax(long minX, long minY, long maxX, long maxY)
        {
            MinX = minX;
            MinY = minY;
            MaxX = maxX;
            MaxY = maxY;
        }

        public void Update(long x, long y)
        {
            MinX = Math.Min(MinX, x);
            MinY = Math.Min(MinY, y);
            MaxX = Math.Max(MaxX, x);
            MaxY = Math.Max(MaxY, y);
        }

        public void Update(MinMax minMax)
        {
            MinX = Math.Min(MinX, minMax.MinX);
            MinY = Math.Min(MinY, minMax.MinY);
            MaxX = Math.Max(MaxX, minMax.MaxX);
            MaxY = Math.Max(MaxY, minMax.MaxY);
        }

        public void Reset()
        {
            MinX = int.MaxValue;
            MinY = int.MaxValue;
            MaxX = int.MinValue;
            MaxY = int.MinValue;
        }
    }
}
