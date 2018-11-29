using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public interface IPhysicsCalc
    {
        MeshPoint[] CurrentMesh { get; }

        void Init();

        void CalcMovement(ref Body[] bodies, float timestep);
    }
}
