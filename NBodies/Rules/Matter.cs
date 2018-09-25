using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace NBodies.Rules
{
    public struct MatterType
    {
        public int Density { get; set; }
        public Color Color { get; set; }

        public MatterType(int density, Color color)
        {
            Density = density;
            Color = color;
        }
    }

    public static class Matter
    {
        public static double Density { get; set; } = 5.0;

        public static MatterType[] Types =
            {
            new MatterType(1,Color.Aqua), // gas
            new MatterType(10, Color.DodgerBlue), // water
            new MatterType(20, Color.Goldenrod), // rock
            new MatterType(30, Color.SaddleBrown), // metal
            new MatterType(60, Color.DarkGray) // heavy metal
            };
    }
}
