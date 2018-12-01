using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public interface IPhysicsCalc
    {
        MeshCell[] CurrentMesh { get; }

        int[,] MeshBodies { get; }

        void Init();

      //  void CalcMovement(ref Body[] bodies, float timestep);

        void CalcMovement(ref Body[] bodies, float timestep, float cellSize);

    }
}
