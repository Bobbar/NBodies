using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public struct GridInfo
    {
        public int OffsetX;
        public int OffsetY;
        public int OffsetZ;
        public int MinX;
        public int MinY;
        public int MinZ;
        public int MaxX;
        public int MaxY;
        public int MaxZ;
        public long Columns;
        public long Rows;
        public long Layers;
        public long Size;
        public long IndexOffset;

        
        public void Set(int offX, int offY, int offZ, long idxOff, int minX, int minY, int minZ, int maxX, int maxY, int maxZ, long cols, long rows, long layers)
        {
            OffsetX = offX;
            OffsetY = offY;
            OffsetZ = offZ;
            IndexOffset = idxOff;
            MinX = minX;
            MinY = minY;
            MinZ = minZ;

            MaxX = maxX;
            MaxY = maxY;
            MaxZ = maxZ;

            Columns = cols;
            Rows = rows;
            Layers = layers;
            Size = cols * rows * layers;
        }
    }
}
