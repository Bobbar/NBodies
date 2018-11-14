using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;


namespace NBodies.IO
{
    public static class Recording
    {
        private static FileStream _recordStream;
        private static int _frameCount = 0;

        public static void StartRecording()
        {
            var dest = new FileInfo(@"C:\Temp\recording.dat");
            _recordStream = dest.Open(FileMode.OpenOrCreate);
           // ProtoBuf.Serializer.Serialize(_recordStream, new Physics.Body[0]);
        }

        public static void StopRecording()
        {
            _recordStream.Close();
        }

        public static void RecordFrame(Physics.Body[] obj)
        {
            ProtoBuf.Serializer.SerializeWithLengthPrefix(_recordStream, obj, ProtoBuf.PrefixStyle.Base128, _frameCount);
            _frameCount++;
        }

        public static void OpenRecording()
        {
            var source = new FileInfo(@"C:\Temp\recording.dat");

            var s = source.OpenRead();

            var data = ProtoBuf.Serializer.Deserialize<Physics.Body[][]>(s);

            Console.WriteLine(data.ToString());
        }

    }
}
