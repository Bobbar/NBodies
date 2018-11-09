using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Drawing2D;


namespace NBodies.Rendering
{
    public class OverlayGraphic
    {
        public OverlayGraphicType Type { get; set; }
        public PointF Location { get; set; }
        public string Value { get; set; }
        public bool Destroyed { get; set; } = false;

        public OverlayGraphic(OverlayGraphicType type, PointF location, string value)
        {
            Type = type;
            Location = location;
            Value = value;
        }

        public void Destroy()
        {
            Destroyed = true;
        }


    }

    public enum OverlayGraphicType
    {
        Text,
        Rect,
        Circle,
        Line
    }
}
