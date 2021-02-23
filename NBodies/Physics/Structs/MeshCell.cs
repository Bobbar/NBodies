using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public struct MeshCell
    {
        public int IdxX;
        public int IdxY;
        public int NeighborStartIdx;
        public int NeighborCount;
        public int BodyStartIdx;
        public int BodyCount;
        public int ChildStartIdx;
        public int ChildCount;
        public float CmX;
        public float CmY;
        public float Mass;
        public int Size;
        public int ParentID;
        public int Level;

        public float LocX
        {
            get
            {
                int exp = (int)Math.Log(Size, 2);
                return (IdxX << exp) + (Size * 0.5f);
            }
        }

        public float LocY
        {
            get
            {
                int exp = (int)Math.Log(Size, 2);
                return (IdxY << exp) + (Size * 0.5f);
            }
        }
    }
}
