using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace NBodies.Rendering
{
    public static class RenderVars
    {
        private static float _currentScale = 1f;
        private static PointF _screenCenter = new PointF();

        public static float CurrentScale
        {
            get
            {
                return _currentScale;
            }

            set
            {
                if (_currentScale != value)
                {
                    _currentScale = value;
                    ScaleOffset = ScaleHelpers.ScalePointExact(ScreenCenter);
                }
            }
        }
        public static PointF ScaleOffset { get; set; } = new PointF();

        public static PointF ViewportOffset = new PointF();
      

        public static PointF ScreenCenter
        {
            get
            {
                return _screenCenter;
            }

            set
            {
                _screenCenter = value;
                ScaleOffset = ScaleHelpers.ScalePointExact(_screenCenter);

            }
        }



    }
}
