using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NBodies.Rendering;

namespace NBodies.Physics
{
   
    public static class PhysicsProvider
    {
        public static IPhysicsCalc PhysicsCalc;

        public static void InitPhysics()
        {
            //PhysicsCalc = new  CUDAFloat(Program.DeviceID, MainLoop.MaxThreadsPerBlock); 
            PhysicsCalc = new OpenCLPhysics(Program.DeviceID, MainLoop.MaxThreadsPerBlock);

            PhysicsCalc.Init();
        }
    }
}
