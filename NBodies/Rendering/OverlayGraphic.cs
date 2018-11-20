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

        public PointF Location2 { get; set; }

        public List<PointF> OrbitPath { get; set; } = new List<PointF>();

        public string Value { get; set; }
        public bool Visible { get; set; } = false;

        public OverlayGraphic(OverlayGraphicType type, PointF location, string value)
        {
            Type = type;
            Location = location;
            Location2 = location;
            Value = value;
        }

        public void Hide()
        {
            Visible = false;
        }

        public void Show()
        {
            Visible = true;
        }


    }

    public enum OverlayGraphicType
    {
        Text,
        Rect,
        Circle,
        Line,
        Orbit
    }
}
