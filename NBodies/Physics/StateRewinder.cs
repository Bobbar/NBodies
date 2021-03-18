using MessagePack;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public class StateRewinder : IDisposable
    {
        public int Position { get { return _position; } }
        public int Count { get { return _maxPosition; } }

        private const int _maxStates = 200;
        private const float _interval = 0.04f;
        private List<byte[]> _states = new List<byte[]>();
        private List<Body[]> _buffer = new List<Body[]>();
        private ManualResetEventSlim _serializerWait = new ManualResetEventSlim(false);
        private ManualResetEventSlim _serializerDoneWait = new ManualResetEventSlim(true);
        private Task _serializerLoop;
        private int _position = -1;
        private float _elap = _interval;

        private int _maxPosition { get { return _states.Count - 1; } }

        public StateRewinder()
        {
            _serializerLoop = new Task(SerializerLoop);
        }

        public void PushState(Body[] frame, float dt)
        {
            if (_serializerLoop.Status != TaskStatus.Running)
                _serializerLoop.Start();

            if (_elap >= _interval)
            {
                var copy = new Body[frame.Length];
                Array.Copy(frame, copy, frame.Length);
                _buffer.Add(copy);
                _serializerWait.Set();
                _elap = 0f;
            }
            _elap += dt;
        }

        private void SerializerLoop()
        {
            while (!disposedValue)
            {
                _serializerWait.Wait();
                _serializerDoneWait.Reset();

                while (_buffer.Count > 0)
                {
                    var state = _buffer.First();

                    if (_position < _maxStates) // Add new state to end.
                    {
                        // If position was moved from the end of the collection, trim off the end.
                        if (_position != -1 && _position < _maxPosition)
                        {
                            _states.RemoveRange(_position + 1, (_maxPosition - _position));
                        }

                        _states.Add(LZ4MessagePackSerializer.Serialize(state));
                        _position++;

                    }
                    else // Remove the first state, then add to end.
                    {
                        _states.RemoveAt(0);
                        _states.Add(LZ4MessagePackSerializer.Serialize(state));
                    }

                    _buffer.RemoveAt(0);
                }

                _serializerWait.Reset();
                _serializerDoneWait.Set();
            }
        }

        public bool TryGetPreviousState(ref Body[] pointer)
        {
            if (!_serializerWait.IsSet)
            {
                if (_position > 0)
                {
                    _position--;
                    pointer = LZ4MessagePackSerializer.Deserialize<Body[]>(_states[_position]);
                    _elap = _interval;
                    return true;
                }
            }
            return false;
        }

        public bool TryGetNextState(ref Body[] pointer)
        {
            if (!_serializerWait.IsSet)
            {
                if (_position + 1 <= _maxPosition)
                {
                    _position++;
                    pointer = LZ4MessagePackSerializer.Deserialize<Body[]>(_states[_position]);
                    return true;
                }
            }

            return false;
        }

        public void Clear()
        {
            _serializerDoneWait.Wait();
            _states.Clear();
            _position = -1;
            _elap = _interval;
        }

        #region IDisposable Support
        private bool disposedValue = false; 

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    _serializerDoneWait.Wait();
                    _states.Clear();
                    _buffer.Clear();
                }
                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
        }
        #endregion


    }
}
