using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public struct LevelInfo
    {
        public SpatialInfo[] Spatials;
        public int[] CellIndex;
        public int CellCount;

        public LevelInfo(SpatialInfo[] spatials, int[] cellIndex, int cellCount)
        {
            Spatials = spatials;
            CellIndex = cellIndex;
            CellCount = cellCount;
        }
    }
}
