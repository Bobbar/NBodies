using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using OpenTK;

namespace NBodies.Shapes
{
    public struct Ellipse
    {
        public PointF Location;
        public float Size;

        public Ellipse(PointF location, float size)
        {
            Location = location;
            Size = size;
        }
    }

    public struct Ellipse3D
    {
        public Vector3 Location;
        public float Size;

        public Ellipse3D(Vector3 location, float size)
        {
            Location = location;
            Size = size;
        }
    }
}
