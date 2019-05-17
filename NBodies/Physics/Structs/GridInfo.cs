using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public struct GridInfo
    {
        public long OffsetX;
        public long OffsetY;
        public long MinX;
        public long MinY;
        public long MaxX;
        public long MaxY;
        public long Columns;
        public long Rows;
        public long Size;
        public long IndexOffset;

        public GridInfo(long offX, long offY, long idxOff, long minX, long minY, long maxX, long maxY, long cols, long rows)
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
