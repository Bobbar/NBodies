using NBodies.Physics;
using NBodies.Rendering;
using System.IO;
using System.Threading;
using System.Windows.Forms;
using ProtoBuf;
using NBodies.Helpers;
using System.Linq;
using System.Collections.Generic;
namespace NBodies.IO
{
    public static class Serializer
    {
        private static string _previousFile = string.Empty;

        public static void SaveState()
        {
            MainLoop.WaitForPause();

            using (var saveDialog = new SaveFileDialog())
            {
                saveDialog.Filter = "NBody State|*.nsta";
                saveDialog.Title = "Save State File";
                saveDialog.ShowDialog();

                if (!string.IsNullOrEmpty(saveDialog.FileName))
                {
                    WriteState(saveDialog.FileName);
                }
            }

            MainLoop.ResumePhysics();
        }

        public static void WriteState(string fileName)
        {
            if (!string.IsNullOrEmpty(fileName))
            {
                using (var fStream = new FileStream(fileName, FileMode.Create))
                {
                    var state = BuildStateParams();
                    ProtoBuf.Serializer.Serialize(fStream, state);
                }
            }
        }

        public static void LoadState()
        {
            MainLoop.WaitForPause();

            using (var openDialog = new OpenFileDialog())
            {
                openDialog.Filter = "NBody State|*.nsta";
                openDialog.Title = "Load State File";
                openDialog.ShowDialog();

                if (!string.IsNullOrEmpty(openDialog.FileName))
                {
                    ReadState(openDialog.FileName);
                }
            }

            MainLoop.ResumePhysics();
        }

        public static void ReadState(string fileName)
        {
            if (!string.IsNullOrEmpty(fileName))
            {
                using (var fStream = new FileStream(fileName, FileMode.Open))
                {
                    _previousFile = fileName;

                    LoadStateStream(fStream);
                }
            }
        }

        public static void LoadPreviousState()
        {
            //_previousFile = $@"C:\Temp\States\TinyTest.nsta";
            //_previousFile = $@"C:\Temp\States\Test2.nsta";
            //  _previousFile = $@"C:\Temp\States\SimpleBlob.nsta";


            if (!string.IsNullOrEmpty(_previousFile))
                ReadState(_previousFile);
        }

        private static void LoadStateStream(Stream stateStream)
        {
            stateStream.Position = 0;
            MainLoop.Stop();

            try
            {
                var state = ProtoBuf.Serializer.Deserialize<StateParams>(stateStream);

                var zmax = state.Bodies.Max(b => b.PosZ);
                if (zmax <= 0)
                {
                    for (int i = 0; i < state.Bodies.Length; i++)
                    {
                        if (!state.Bodies[i].HasFlag(Flags.BlackHole))
                        {
                            float rndZ = Numbers.GetRandomFloat(-1.0f, 1.0f);
                            state.Bodies[i].PosZ = rndZ;
                        }
                    }
                }


                LoadStateParams(state);
            }
            catch // Try to load an old style state.
            {
                stateStream.Position = 0;
                var bodies = ProtoBuf.Serializer.Deserialize<Body[]>(stateStream);

                var zmax = bodies.Max(b => b.PosZ);
                if (zmax <= 0)
                {
                    for (int i = 0; i < bodies.Length; i++)
                    {
                        if (!bodies[i].HasFlag(Flags.BlackHole))
                        {
                            float rndZ = Numbers.GetRandomFloat(-10.0f, 10.0f);
                            bodies[i].PosZ = rndZ;
                        }
                    }
                }

                BodyManager.ReplaceBodies(bodies);
            }

            MainLoop.StartLoop();
        }

        private static StateParams BuildStateParams()
        {
            var settings = MainLoop.GetSettings();
            var state = new StateParams();
            state.KernelSize = settings.KernelSize;
            state.DeltaTime = settings.DeltaTime;
            state.Viscosity = settings.Viscosity;
            state.GasK = settings.GasK;
            state.MeshLevels = settings.MeshLevels;
            state.CellSizeExponent = settings.CellSizeExponent;
            state.Bodies = BodyManager.Bodies;
            return state;
        }

        private static void LoadStateParams(StateParams state)
        {
            MainLoop.KernelSize = state.KernelSize;
            MainLoop.TimeStep = state.DeltaTime;
            MainLoop.Viscosity = state.Viscosity;
            MainLoop.GasK = state.GasK;
            MainLoop.MeshLevels = state.MeshLevels;
            MainLoop.CellSizeExp = state.CellSizeExponent;
            BodyManager.ReplaceBodies(state.Bodies);
        }
    }

    [ProtoContract]
    public class StateParams
    {
        [ProtoMember(1)]
        public float KernelSize { get; set; }
        [ProtoMember(2)]
        public float DeltaTime { get; set; }
        [ProtoMember(3)]
        public float Viscosity { get; set; }
        [ProtoMember(4)]
        public float GasK { get; set; }
        [ProtoMember(5)]
        public int MeshLevels { get; set; }
        [ProtoMember(6)]
        public int CellSizeExponent { get; set; }

        [ProtoMember(7)]
        public Body[] Bodies { get; set; }

        public StateParams() { }

        public StateParams(Body[] bodies)
        {
            Bodies = bodies;
        }
    }
}