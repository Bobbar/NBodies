using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public struct SpatialInfo
    {
        public int Mort;
        public int IdxX;
        public int IdxY;
        public int BodyIdx;

        public SpatialInfo(int mort, int idX, int idY, int bodyIdx)
        {
            Mort = mort;
            IdxX = idX;
            IdxY = idY;
            BodyIdx = bodyIdx;
        }
    }
}
