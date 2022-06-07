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
        private static Cloo.ComputeDevice _computeDevice;

        public static void InitPhysics()
        {
            PhysicsCalc = new OpenCLPhysics(Program.DeviceID, MainLoop.MaxThreadsPerBlock);

            PhysicsCalc.Init();
        }

        public static void InitPhysics(Cloo.ComputeDevice device, int threadsPerBlock, bool fastMath)
        {
            _computeDevice = device;

            MainLoop.MaxThreadsPerBlock = threadsPerBlock;
            PhysicsCalc = new OpenCLPhysics(device, MainLoop.MaxThreadsPerBlock, fastMath);

            PhysicsCalc.Init();
        }

        public static void Reload()
        {
            MainLoop.WaitForPause();

            PhysicsCalc.Dispose();
            PhysicsCalc = new OpenCLPhysics(_computeDevice, MainLoop.MaxThreadsPerBlock);
            PhysicsCalc.Init();
        }
    }
}
