﻿using System.Drawing;

namespace NBodies.Helpers
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
                    ScaleOffset = ScaleHelpers.FieldPointToScreenUnscaled(ScreenCenter);
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
                ScaleOffset = ScaleHelpers.FieldPointToScreenUnscaled(_screenCenter);
            }
        }
    }
}