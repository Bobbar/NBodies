using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;

namespace NBodies.Physics
{
    [Cudafy(eCudafyType.Struct)]
    public struct MeshPoint
    {
        public float LocX;
        public float LocY;
        public float CmX;
        public float CmY;
        public double Mass;
        public int Count;
        public float Size;
        public float Top;
        public float Bottom;
        public float Left;
        public float Right;

    }
}
