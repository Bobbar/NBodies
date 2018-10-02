﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies
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
}
