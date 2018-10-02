using NBodies.Rendering;
using System.IO;
using System.Threading;
using System.Windows.Forms;
using NBodies.Physics;

namespace NBodies.IO
{
    public static class Serializer
    {
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
                    using (var fStream = new FileStream(saveDialog.FileName, FileMode.OpenOrCreate))
                    {
                        ProtoBuf.Serializer.Serialize(fStream, BodyManager.Bodies);
                    }
                }
            }

            MainLoop.Resume();
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
                    using (var fStream = new FileStream(openDialog.FileName, FileMode.Open))
                    {
                        MainLoop.Stop();
                        Thread.Sleep(200);
                        BodyManager.ReplaceBodies(ProtoBuf.Serializer.Deserialize<Body[]>(fStream));
                        MainLoop.StartLoop();
                    }
                }
            }

            MainLoop.Resume();
        }
    }
}