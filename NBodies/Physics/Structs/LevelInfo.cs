﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

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

        public LevelInfo(int[] cellIndex, int cellCount)
        {
            Spatials = new SpatialInfo[0];
            CellIndex = cellIndex;
            CellCount = cellCount;
        }
    }
}
