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
        public int MinX;
        public int MinY;
        public int MaxX;
        public int MaxY;
        public int Columns;
        public int Rows;
        public int Size;
        public int IndexOffset;

        public GridInfo(int offX, int offY, int idxOff, int minX, int minY, int maxX, int maxY, int cols, int rows)
        {
            OffsetX = offX;
            OffsetY = offY;
            IndexOffset = idxOff;
            MinX = minX;
            MinY = minY;
            MaxX = maxX;
            MaxY = maxY;
            Columns = cols;
            Rows = rows;

            Size = ((cols + 1) * (rows + 1));
        }
    }
}
