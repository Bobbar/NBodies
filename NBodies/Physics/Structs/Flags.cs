using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public enum Flags
    {
        BlackHole = 1,
        IsExplosion = 2,
        Culled = 4,
        InRoche = 8
    }
}
