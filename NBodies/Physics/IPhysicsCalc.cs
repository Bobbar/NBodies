using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public interface IPhysicsCalc : IDisposable
    {
        MeshCell[] CurrentMesh { get; }

        int[] LevelIndex { get; }

        void Init();

        void CalcMovement(ref Body[] bodies, SimSettings sim, int threadsPerBlock, long bufferVersion, out bool postNeeded);

        void FixOverLaps(ref Body[] bodies);

        void Flush();
    }
}
