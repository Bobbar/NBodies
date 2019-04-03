using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public struct MinMax
    {
        public int MinX;
        public int MinY;
        public int MaxX;
        public int MaxY;

        public MinMax(int dummy = 0)
        {
            MinX = int.MaxValue;
            MinY = int.MaxValue;
            MaxX = int.MinValue;
            MaxY = int.MinValue;
        }

        public MinMax(int minX, int minY, int maxX, int maxY)
        {
            MinX = minX;
            MinY = minY;
            MaxX = maxX;
            MaxY = maxY;
        }


        public void Update(int X, int Y)
        {
            MinX = Math.Min(MinX, X);
            MinY = Math.Min(MinY, Y);
            MaxX = Math.Max(MaxX, X);
            MaxY = Math.Max(MaxY, Y);
        }

        public void Update(MinMax minMax)
        {
            MinX = Math.Min(MinX, minMax.MinX);
            MinY = Math.Min(MinY, minMax.MinY);
            MaxX = Math.Max(MaxX, minMax.MaxX);
            MaxY = Math.Max(MaxY, minMax.MaxY);
        }
    }
}
