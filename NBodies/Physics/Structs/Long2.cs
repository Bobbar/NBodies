using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public struct uint2
    {
        //long X;
        //long Y;

        //public void Set(long x, long y) { X = x; Y = y; }

        //int X;
        //int Y;

        //public void Set(int x, int y) { X = x; Y = y; }

        uint X;
        uint Y;

        public void Set(uint x, uint y) { X = x; Y = y; }

        public override string ToString()
        {
            return $"{X}, {Y}";
        }
    }

    public struct int2
    {
        public int X;
        public int Y;

        public void Set(int x, int y) { X = x; Y = y; }

        public override string ToString()
        {
            return $"{X}, {Y}";
        }
    }

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
}
