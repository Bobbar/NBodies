using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace NBodies.Physics
{
    public struct SimSettings
    {
        public float DeltaTime;
        public float Viscosity;
        public float CullDistance;
        public int CollisionsOn;
        public int MeshLevels;
        public int CellSizeExponent;
    }
}
