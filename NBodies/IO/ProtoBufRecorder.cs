using NBodies.Physics;
using ProtoBuf;
using System;
using System.Collections.Generic;
using System.IO;

namespace NBodies.IO
{
    public class ProtoBufRecorder : IRecording
    {
        private Stream _stream;
        private bool _playbackComplete = false;
        private int _frameCount = 0;
        private int _currentFrameIdx = 0;
        private bool _playbackPaused = false;
        private bool _playbackActive = false;
        private bool _recordingActive = false;
        private long[] _frameIndex = new long[0];
        private Body[] _currentFrame = new Body[0];
        private int _seekIndex = -1;
        private object _lockObject = new object();

        private PrefixStyle _prefixStyle = PrefixStyle.Fixed32;

        public event EventHandler<int> ProgressChanged;

        public int SeekIndex
        {
            get
            {
                return _seekIndex;
            }

            set
            {
                if (_seekIndex != value)
                {
                    _seekIndex = value;
                    SetCurrentFrame(_seekIndex);
                }
            }
        }

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

        public bool PlaybackActive
        {
            get { return _playbackActive; }
        }

        public int TotalFrames
        {
            get { return _frameCount; }
        }

        public bool PlaybackComplete
        {
            get { return _playbackComplete; }
        }

        public bool RecordingActive
        {
            get { return _recordingActive; }
        }

        public double FileSize
        {
            get
            {
                if (_stream != null)
                {
                    return _stream.Length;
                }
                return 0;
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

            lock (_lockObject)
            {
                _stream = dest.Open(FileMode.OpenOrCreate);
                _stream.Position = 0;
            }

            _recordingActive = true;
        }

        public void CreateRecording(Stream stream)
        {
            StopAll();

            lock (_lockObject)
            {
                _stream = stream;
                _stream.Position = 0;
            }

            _recordingActive = true;
        }

        private void SetCurrentFrame(int frameIndex)
        {
            _currentFrameIdx = frameIndex;
            _currentFrame = GetFrame(frameIndex);
        }

        // Returns the frame at the specified index.
        private Body[] GetFrame(int index)
        {
            lock (_lockObject)
            {
                _stream.Position = _frameIndex[index];
                return ProtoBuf.Serializer.DeserializeWithLengthPrefix<Body[]>(_stream, _prefixStyle, 0);
            }
        }

        public Body[] GetFrameAtPosition(long pos)
        {
            lock (_lockObject)
            {
                _stream.Position = pos;
                return ProtoBuf.Serializer.DeserializeWithLengthPrefix<Body[]>(_stream, _prefixStyle, 0);
            }
        }

        public Body[] GetNextFrame()
        {
            Body[] newFrame;

            if (_stream == null || !_stream.CanRead)
                return null;

            // If paused, return the current frame.
            if (_playbackPaused)
            {
                return _currentFrame;
            }
            else // Otherwise, get the next frame in the stream.
            {
                lock (_lockObject)
                {
                    newFrame = ProtoBuf.Serializer.DeserializeWithLengthPrefix<Body[]>(_stream, _prefixStyle, 0);
                }

                _currentFrame = newFrame;

                // Pause playback if we are at the end of the stream.
                if (newFrame == null || newFrame.Length < 1)
                {
                    _playbackPaused = true;
                }
                else
                {
                    _currentFrameIdx++;
                    OnProgressChanged(_currentFrameIdx);
                }

                return newFrame;
            }
        }

        public void OpenRecording(string file)
        {
            StopAll();

            var source = new FileInfo(file);
            _stream = source.OpenRead();
            _stream.Position = 0;

            _currentFrameIdx = 0;

            GetFrameCount();

            _playbackActive = true;
            _playbackPaused = false;
            _playbackComplete = false;
        }

        public void RecordFrame(Body[] frame)
        {
            lock (_lockObject)
            {
                ProtoBuf.Serializer.SerializeWithLengthPrefix(_stream, frame, _prefixStyle, 0);
            }
        }

        public void StopAll()
        {
            _playbackActive = false;
            _playbackComplete = true;
            _recordingActive = false;
            _currentFrame = new Body[0];
            _currentFrameIdx = 0;
            _stream?.Close();
        }

        private void GetFrameCount()
        {
            int count = 0;
            int len = 0;

            // List which contains the positions of each frame.
            // Used for fast lookups when seeking.
            var frameIdxList = new List<long>();

            while (ProtoBuf.Serializer.TryReadLengthPrefix(_stream, _prefixStyle, out len))
            {
                count++;
                _stream.Seek(len, SeekOrigin.Current);
                frameIdxList.Add(_stream.Position);
            }

            _stream.Position = 0;

            _frameIndex = frameIdxList.ToArray();
           
            _frameCount = count - 1;
        }
       
    }
}