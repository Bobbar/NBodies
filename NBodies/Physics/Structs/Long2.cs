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
}
