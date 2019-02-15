using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;

namespace NBodies.Physics
{
    [Cudafy(eCudafyType.Struct)]
    public struct MeshCell
    {
        public int ID;
        public float LocX;
        public float LocY;
        public int xID;
        public int yID;
        public int Mort;
        public float CmX;
        public float CmY;
        public double Mass;
        public int Size;
        public int BodyStartIdx;
        public int BodyCount;
        public int NeighborStartIdx;
        public int NeighborCount;
        public int ParentID;
        public int Level;
        public int ChildIdxStart;
        public int ChildCount;
    }
}
