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

    public struct int4
    {
        public int X;
        public int Y;
        public int Z;
        public int W;
    }

    public struct float2
    {
        public float X;
        public float Y;
    }

    public struct float4
    {
        public float X;
        public float Y;
        public float Z;
        public float W;
    }
}
