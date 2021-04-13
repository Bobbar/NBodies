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
        public float Density { get; set; }
        public Color Color { get; set; }
        /// <summary>
        /// Percentage indicating how often this matter type will be returned with GetRandom().
        /// 
        /// All defined types must sum to 100.
        /// </summary>
        public int Occurrence { get; set; }

        public MatterType(int density, Color color)
        {
            Density = density;
            Color = color;
            Occurrence = 0;
        }

        public MatterType(int density, Color color, int occurrence)
        {
            Density = density;
            Color = color;
            Occurrence = occurrence;
        }
    }

    public static class Matter
    {
        private static Random _rnd = new Random((int)(DateTime.Now.Ticks % int.MaxValue));

        public static float Density { get; set; } = 1.0f;

        public static MatterType[] Types =
        {
            new MatterType(2,Color.Aqua, 44), // gas
            new MatterType(10, Color.DodgerBlue, 43), // water
            new MatterType(20, Color.Goldenrod, 5), // rock
            new MatterType(30, Color.SaddleBrown,4), // metal
            new MatterType(60, Color.DarkGray, 4) // heavy metal
        };

        public static MatterType GetRandom()
        {
            Range[] matterRanges = new Range[Types.Length];
            int position = 0;

            for (int i = 0; i < Types.Length; i++)
            {
                Range range = new Range();
                range.Index = i;
                range.Start = position;
                range.End = position + Types[i].Occurrence;

                matterRanges[i] = range;

                position += Types[i].Occurrence;
            }

            int select = _rnd.Next(0, 100 + 1);

            for (int i = 0; i < matterRanges.Length; i++)
            {
                Range range = matterRanges[i];

                if (select > range.Start && select < range.End)
                {
                    return Types[range.Index];
                }
            }

            return Types[0];
        }

        public static MatterType GetForDistance(float dist, float max)
        {
            int layers = Types.Length;
            float layerSize = max / layers;
            var sortMatter = Types.OrderByDescending(m => m.Density).ToArray();

            int layer = (int)(dist / layerSize);

            return sortMatter[layer];
        }

    }

    public struct Range
    {
        public int Index { get; set; }
        public int Start { get; set; }
        public int End { get; set; }

    }
}
