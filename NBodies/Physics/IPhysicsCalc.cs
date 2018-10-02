using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public interface IPhysicsCalc
    {
        void Init();

        void CalcMovement(Body[] bodies, float timestep);
    }
}
