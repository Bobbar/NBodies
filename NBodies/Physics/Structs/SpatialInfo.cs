﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public struct SpatialInfo
    {
        public long Mort;
        public int IdxX;
        public int IdxY;
        public int IdxZ;
        public int Index;

        public SpatialInfo(long mort, int idX, int idY, int index)
        {
            Mort = mort;
            IdxX = idX;
            IdxY = idY;
            IdxZ = 0;
            Index = index;
        }

        public void Set(long mort, int idX, int idY, int index)
        {
            Mort = mort;
            IdxX = idX;
            IdxY = idY;
            Index = index;
        }

        public void Set(long mort, int idX, int idY, int idZ, int index)
        {
            Mort = mort;
            IdxX = idX;
            IdxY = idY;
            IdxZ = idZ;
            Index = index;
        }
    }
}
