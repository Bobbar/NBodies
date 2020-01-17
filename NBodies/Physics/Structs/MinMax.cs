using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public class MinMax
    {
        public int MinX;
        public int MinY;
        public int MinZ;

        public int MaxX;
        public int MaxY;
        public int MaxZ;

        public MinMax()
        {
            MinX = int.MaxValue;
            MinY = int.MaxValue;
            MinZ = int.MaxValue;

            MaxX = int.MinValue;
            MaxY = int.MinValue;
            MaxZ = int.MinValue;

        }

        public MinMax(int minX, int minY, int maxX, int maxY)
        {
            MinX = minX;
            MinY = minY;
            MinZ = 0;
            MaxX = maxX;
            MaxY = maxY;
            MaxZ = 0;
        }

        public MinMax(int minX, int minY, int minZ, int maxX, int maxY, int maxZ)
        {
            MinX = minX;
            MinY = minY;
            MinZ = minZ;
            MaxX = maxX;
            MaxY = maxY;
            MaxZ = maxZ;
        }

        public void Update(int x, int y)
        {
            MinX = Math.Min(MinX, x);
            MinY = Math.Min(MinY, y);
            MaxX = Math.Max(MaxX, x);
            MaxY = Math.Max(MaxY, y);
        }

        public void Update(int x, int y, int z)
        {
            MinX = Math.Min(MinX, x);
            MinY = Math.Min(MinY, y);
            MinZ = Math.Min(MinZ, z);

            MaxX = Math.Max(MaxX, x);
            MaxY = Math.Max(MaxY, y);
            MaxZ = Math.Max(MaxZ, z);

        }

        public void Update(MinMax minMax)
        {
            MinX = Math.Min(MinX, minMax.MinX);
            MinY = Math.Min(MinY, minMax.MinY);
            MinZ = Math.Min(MinZ, minMax.MinZ);

            MaxX = Math.Max(MaxX, minMax.MaxX);
            MaxY = Math.Max(MaxY, minMax.MaxY);
            MaxZ = Math.Max(MaxZ, minMax.MaxZ);

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
