using System.Drawing;

namespace NBodies.Helpers
{
    public static class ViewportOffsets
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
                    ScaleOffset = ViewportHelpers.FieldPointToScreenNoOffset(ScreenCenter);
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
                ScaleOffset = ViewportHelpers.FieldPointToScreenNoOffset(_screenCenter);
            }
        }

        /// <summary>
        /// Displaces the viewport offset to zoom in towards the specified location.
        /// </summary>
        /// <param name="newScale">New scale to be applied.</param>
        /// <param name="location">Screen location to zoom towards.</param>
        public static void Zoom(float newScale, Point location)
        {
            float factor = 0.05f;   // Displacement factor.
            bool zoomIn = true;     // Determines displacement direction.

            if (_currentScale < newScale)
                zoomIn = false;

            // Set the new scale.
            CurrentScale = newScale;

            // Determine scaled distance from screen center.
            var d = ViewportHelpers.FieldPointToScreenNoOffset(new PointF((location.X - (_screenCenter.X)), (location.Y - (_screenCenter.Y))));

            // Apply the distplacement factor.
            var dx = d.X * (factor);
            var dy = d.Y * (factor);

            // Flip the displacment direction as needed.
            if (zoomIn)
            {
                dx *= -1;
                dy *= -1;
            }

            // Apply the final displacement to the viewport offset.
            ViewportOffset = new PointF(ViewportOffset.X - dx, ViewportOffset.Y - dy);
        }
    }
}