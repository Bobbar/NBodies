using System;
using System.Collections.Generic;

namespace NBodies.Helpers
{
    public static class Numbers
    {
        private static Random _rnd = new Random((int)(DateTime.Now.Ticks % Int32.MaxValue));

        public static float GetRandomFloat(float min, float max)
        {
            float range = max - min;
            float sample = (float)_rnd.NextDouble();
            float scaled = (sample * range) + min;
            return scaled;
        }

        public static int GetRandomInt(int min, int max)
        {
            return _rnd.Next(min, max + 1);
        }
    }

    /// <summary>
    /// Provides a round-robin running average for a float value over time.
    /// </summary>
    public class Average
    {
        private List<float> _values = new List<float>();
        private int _max;
        private int _position = 0;
        private float _current;

        /// <summary>
        /// Current average.
        /// </summary>
        public float Current
        {
            get
            {
                return _current;
            }
        }

        /// <summary>
        /// Creates a new instance with the specified max number of values.
        /// </summary>
        /// <param name="max">The max number of values to maintain an average of.</param>
        public Average(int max)
        {
            _max = max;
        }

        /// <summary>
        /// Add a new value to the averaged collection.
        /// </summary>
        /// <param name="value">Value to be added to the collection and averaged.</param>
        /// <returns>Returns the new accumulative average value.</returns>
        public float Add(float value)
        {
            // Reset the position if we reach the end of the collection.
            if (_position >= _max)
                _position = 0;

            // Add new values until the collection is full, then do round robin.
            if (_values.Count < _max)
            {
                _values.Add(value);
            }
            else
            {
                _values[_position] = value;
            }

            // Sum all values and compute the average.
            double total = 0;
            for (int i = 0; i < _values.Count; i++)
            {
                total += _values[i];
            }

            _current = (float)total / _values.Count;

            // Move to next position.
            _position++;

            return _current;
        }
    }
}