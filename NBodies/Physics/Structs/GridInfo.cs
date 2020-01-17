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

        public int Columns;
        public int Rows;
        public int Layers;
        public int Size;
        public int IndexOffset;

        public GridInfo(int offX, int offY, int idxOff, int minX, int minY, int maxX, int maxY, int cols, int rows)
        {
            OffsetX = offX;
            OffsetY = offY;
            OffsetZ = 0;
            IndexOffset = idxOff;
            MinX = minX;
            MinY = minY;
            MinZ = 0;
            MaxX = maxX;
            MaxY = maxY;
            MaxZ = 0;
            Columns = cols;
            Rows = rows;
            Layers = 0;

            Size = ((cols + 1) * (rows + 1));
        }

        public void Set(int offX, int offY, int idxOff, int minX, int minY, int maxX, int maxY, int cols, int rows)
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

        public void Set(int offX, int offY, int offZ, int idxOff, int minX, int minY, int minZ, int maxX, int maxY, int maxZ, int cols, int rows, int layers)
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
            Size = ((cols + 1) * (rows + 1) * (layers + 1));
        }
    }
}
