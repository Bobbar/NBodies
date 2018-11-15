﻿using MessagePack;
using NBodies.Physics;
using System;
using System.Collections.Generic;
using System.IO;

namespace NBodies.IO
{
    public class MessagePackRecorder : IRecording
    {
        private FileStream _fileStream;
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
                _fileStream = dest.Open(FileMode.OpenOrCreate);
                _fileStream.Position = 0;
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
                _fileStream.Position = _frameIndex[index];
                return LZ4MessagePackSerializer.Deserialize<Body[]>(_fileStream, true);
            }
        }

        public Body[] GetNextFrame()
        {
            Body[] newFrame = new Body[0];

            if (_fileStream == null || !_fileStream.CanRead)
                return null;

            // If paused, return the current frame.
            if (_playbackPaused)
            {
                return _currentFrame;
            }
            else // Otherwise, get the next frame in the stream.
            {
                if (_fileStream.Position < _fileStream.Length)
                {
                    lock (_lockObject)
                    {
                        newFrame = LZ4MessagePackSerializer.Deserialize<Body[]>(_fileStream, true);
                    }
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
            _fileStream = source.OpenRead();
            _fileStream.Position = 0;

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
                LZ4MessagePackSerializer.Serialize(_fileStream, frame);
            }
        }

        public void StopAll()
        {
            _playbackActive = false;
            _playbackComplete = true;
            _recordingActive = false;
            _currentFrame = new Body[0];
            _currentFrameIdx = 0;
            _fileStream?.Close();
        }

        private void GetFrameCount()
        {
            int count = 0;

            // List which contains the positions of each frame.
            // Used for fast lookups when seeking.
            var frameIdxList = new List<long>();

            while (_fileStream.Position < _fileStream.Length)
            {
                frameIdxList.Add(_fileStream.Position);
                MessagePackBinary.ReadNextBlock(_fileStream);
                count++;
            }

            _fileStream.Position = 0;

            _frameIndex = frameIdxList.ToArray();

            _frameCount = count - 1;
        }
    }
}