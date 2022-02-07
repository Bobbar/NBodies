using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK;

namespace NBodies.Physics
{
    public struct MeshCell
    {
        public int IdxX;
        public int IdxY;
        public int IdxZ;
        public int NeighborStartIdx;
        public int NeighborCount;
        public int BodyStartIdx;
        public int BodyCount;
        public int ChildStartIdx;
        public int ChildCount;
        public float Mass;
        public float CmX;
        public float CmY;
        public float CmZ;
        public int Size;
        public int ParentID;

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

        public float LocZ
        {
            get
            {
                int exp = (int)Math.Log(Size, 2);
                return (IdxZ << exp) + (Size * 0.5f);
            }
        }

        public Vector3 PositionVec()
        {
            return new Vector3(LocX, LocY, LocZ);
        }

    }
}
