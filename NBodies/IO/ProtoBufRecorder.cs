using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NBodies.Physics;
using System.IO;
using ProtoBuf;

namespace NBodies.IO
{


    public class ProtoBufRecorder : IRecording
    {
        private FileStream _fileStream;
        private bool _playbackComplete = true;
        private int _frameCount = 0;
        private int _currentFrame = 0;
        private bool _playbackPaused = false;

        private PrefixStyle _prefixStyle = PrefixStyle.Fixed32;//PrefixStyle.Base128;

        public event EventHandler<int> ProgressChanged;

        public bool PlaybackPaused
        {
            get
            {
                return _playbackPaused;
            }

            set
            {
                _playbackPaused = value;
            }
        }

        public int TotalFrames
        {
            get { return _frameCount; }
        }

        public bool PlaybackComplete
        {
            get
            {
                return _playbackComplete;
            }
        }

        private void OnProgressChanged(int position)
        {
            ProgressChanged?.Invoke(this, position);
        }

        public void CreateRecording(string file)
        {
            StopAll();

            var dest = new FileInfo(file);
            _fileStream = dest.Open(FileMode.OpenOrCreate);
            _fileStream.Position = 0;
        }

        public Body[] GetNextFrame()
        {
            if (_fileStream == null || !_fileStream.CanRead)
                return null;

            var frame = ProtoBuf.Serializer.DeserializeWithLengthPrefix<Body[]>(_fileStream, _prefixStyle, 0);

            if (frame == null || frame.Length < 1)
            {
                _playbackPaused = true;
            }
            else
            {
                _currentFrame++;
                OnProgressChanged(_currentFrame);
            }

            return frame;
        }

        public void OpenRecording(string file)
        {
            StopAll();

            var source = new FileInfo(file);
            _fileStream = source.OpenRead();
            _fileStream.Position = 0;

            _currentFrame = 0;

            GetFrameCount();

            _playbackPaused = false;
            _playbackComplete = false;
        }

        public void RecordFrame(Body[] frame)
        {
            ProtoBuf.Serializer.SerializeWithLengthPrefix(_fileStream, frame, _prefixStyle, 0);
        }

        public void StopAll()
        {
            _currentFrame = 0;
            _fileStream?.Close();
        }

        private void GetFrameCount()
        {
            int count = 0;
            int len = 0;

            while (ProtoBuf.Serializer.TryReadLengthPrefix(_fileStream, _prefixStyle, out len))
            {
                count++;
               _fileStream.Seek(len, SeekOrigin.Current);
            }

            _fileStream.Position = 0;

            _frameCount = count;
        }

        public void SeekToFrame(int frame)
        {
            _fileStream.Position = 0;
            _currentFrame = 0;

            for (int i = 0; i < frame; i++)
            {
                int len = 0;
                if (ProtoBuf.Serializer.TryReadLengthPrefix(_fileStream, _prefixStyle, out len))
                {
                    _currentFrame++;
                    _fileStream.Seek(len, SeekOrigin.Current);
                }
            }

        }
    }
}
