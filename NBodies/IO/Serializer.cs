using NBodies.Physics;
using NBodies.Rendering;
using System.IO;
using System.Threading;
using System.Windows.Forms;

namespace NBodies.IO
{
    public static class Serializer
    {
        private static Stream _previousStream = new MemoryStream();

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

            MainLoop.Resume();
        }

        public static void WriteState(string fileName)
        {
            if (!string.IsNullOrEmpty(fileName))
            {
                using (var fStream = new FileStream(fileName, FileMode.Create))
                {
                    ProtoBuf.Serializer.Serialize(fStream, BodyManager.Bodies);
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

            MainLoop.Resume();
        }

        public static void ReadState(string fileName)
        {
            if (!string.IsNullOrEmpty(fileName))
            {
                using (var fStream = new FileStream(fileName, FileMode.Open))
                {
                    _previousStream = new MemoryStream();

                    fStream.CopyTo(_previousStream);

                    LoadStateStream(fStream);
                }
            }
        }

        public static void LoadPreviousState()
        {
            if (_previousStream != null && _previousStream.Length > 0)
            {
                LoadStateStream(_previousStream);
            }
        }

        private static void LoadStateStream(Stream stateStream)
        {
            stateStream.Position = 0;
            MainLoop.Stop();
            BodyManager.ReplaceBodies(ProtoBuf.Serializer.Deserialize<Body[]>(stateStream));
            MainLoop.StartLoop();
        }
    }
}