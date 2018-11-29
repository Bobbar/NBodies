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
        public float Mass;
        public int Count;
    }
}
