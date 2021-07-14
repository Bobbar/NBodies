using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public struct long2
    {
        public long X;
        public long Y;

        public void Set(long x, long y) { X = x; Y = y; }

        public override string ToString()
        {
            return $"{X}, {Y}";
        }
    }

    public struct int2
    {
        public int X;
        public int Y;
    }

    public struct int3
    {
        public int X;
        public int Y;
        public int Z;
        public int dummy; // Needed because cl_int3 is 16 bytes wide.
    }

    public struct float3
    {
        public float X;
        public float Y;
        public float Z;
        public float dummy; // Needed because cl_float3 is 16 bytes wide.
    }
}
