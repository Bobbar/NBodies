using NBodies.Physics;
using System;

namespace NBodies.IO
{
    public interface IRecording
    {
        event EventHandler<int> ProgressChanged;

        bool PlaybackActive { get; }
        bool PlaybackComplete { get; }
        bool PlaybackPaused { get; set; }
        bool RecordingActive { get; }

        int SeekIndex { get; set; }
        int TotalFrames { get; }

        double FileSize { get; }

        void CreateRecording(string file);

        Body[] GetNextFrame();

        void OpenRecording(string file);

        void RecordFrame(Body[] frame);

        void StopAll();
    }
}