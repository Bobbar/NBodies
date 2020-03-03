using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Rendering
{
    public static class RenderVars
    {
        public static int PointSpriteTexIdx = 0;

        public static List<OverlayGraphic> OverLays = new List<OverlayGraphic>();
        public static bool AAEnabled = true;
        public static bool Trails = false;
        public static bool ClipView = true;
        public static bool ShowForce = false;
        public static bool ShowAllForce = false;
        public static bool ShowPath = false;
        public static bool ShowMesh = false;
        public static bool SortZOrder = true;
        public static bool FastPrimitives = true;
        public static DisplayStyle DisplayStyle = DisplayStyle.Normal;

        public static float StyleScaleMax
        {
            get
            {
                return _styleScales[(int)DisplayStyle];
            }

            set
            {
                if (value > 0 && value <= 8000)
                {
                    _styleScales[(int)DisplayStyle] = value;
                }
            }
        }

        public static int BodyAlpha
        {
            get
            {
                return _bodyAlpha;
            }

            set
            {
                if (value >= 0 && value <= 255)
                {
                    _bodyAlpha = value;
                }
            }
        }

        private static float _styleScaleMax = 210;
        private static int _bodyAlpha = 210;
        private static float[] _styleScales = new float[1] { _styleScaleMax };

        public static void SetStyleScales()
        {
            int styleCount = Enum.GetValues(typeof(DisplayStyle)).Cast<int>().Max() + 1;
            _styleScales = new float[styleCount];
            for (int i = 0; i < _styleScales.Length; i++)
                _styleScales[i] = _styleScaleMax;
        }

        public static void AddOverlay(OverlayGraphic overlay)
        {
            if (!OverLays.Contains(overlay))
            {
                OverLays.Add(overlay);
            }
        }
    }
}
