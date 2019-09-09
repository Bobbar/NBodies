using System;
using System.Drawing;

namespace NBodies.Helpers
{
    public static class ColorHelper
    {
        private static Random rnd = new Random((int)DateTime.Now.Ticks);

        public static Color RandomColor()
        {
            return Color.FromArgb(255, rnd.Next(0, 255), rnd.Next(0, 255), rnd.Next(0, 255));
        }
    }
}