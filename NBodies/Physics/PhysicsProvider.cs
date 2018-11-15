using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
   
    public static class PhysicsProvider
    {
        public static IPhysicsCalc PhysicsCalc;

        public static void InitPhysics()
        {
            PhysicsCalc = new  CUDAFloat(Program.DeviceID, Program.ThreadsPerBlockArgument); //CUDADouble(2); 
            PhysicsCalc.Init();
        }
    }
}
