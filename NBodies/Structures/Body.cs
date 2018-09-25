using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System.Drawing;


namespace NBodies.Structures
{
    [Cudafy]
    public struct Body
    {
        public double LocX { get; set; }
        public double LocY { get; set; }
        public double Mass { get; set; }
        public double SpeedX { get; set; }
        public double SpeedY { get; set; }
        public double ForceX { get; set; }
        public double ForceY { get; set; }
        public double ForceTot { get; set; }
        public int Color { get; set; }
        public double Size { get; set; }
        public int Visible { get; set; }
        public int InRoche { get; set; }
        public int BlackHole { get; set; }
        public int UID { get; set; }

      

       

    }
}
