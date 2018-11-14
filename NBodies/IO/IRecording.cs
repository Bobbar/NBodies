using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using NBodies.Physics;


namespace NBodies.IO
{
    public interface IRecording
    {
        event EventHandler<int> ProgressChanged;

        bool PlaybackPaused { get; set; }

        int TotalFrames { get; }

        void CreateRecording(string file);

        void StopAll();

        void RecordFrame(Body[] frame);

        void OpenRecording(string file);

        Body[] GetNextFrame();

        bool PlaybackComplete { get; }

        void SeekToFrame(int frame);

    }
}
