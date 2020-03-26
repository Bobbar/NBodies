using NBodies.Extensions;
using NBodies.Physics;
using System.Drawing;
using OpenTK;

namespace NBodies.Helpers
{
    public static class ViewportHelpers
    {
        public static Vector3 CameraPos = new Vector3();
        public static Vector3 CameraDirection = new Vector3();

        public static PointF ScreenPointToField(PointF point)
        {
            PointF convertedPos = new PointF((point.X / ViewportOffsets.CurrentScale) - ViewportOffsets.ViewportOffset.X - ViewportOffsets.ScaleOffset.X, (point.Y / ViewportOffsets.CurrentScale) - ViewportOffsets.ViewportOffset.Y - ViewportOffsets.ScaleOffset.Y);

            return convertedPos;
        }

        public static PointF FieldPointToScreen(PointF point)
        {
            PointF convertedPos = point.Add(ViewportOffsets.ViewportOffset.Add(ViewportOffsets.ScaleOffset)).Multi(ViewportOffsets.CurrentScale);

            return convertedPos;
        }

        public static PointF FieldPointToScreenNoOffset(PointF point)
        {
            PointF convertedPos = new PointF((point.X / ViewportOffsets.CurrentScale), (point.Y / ViewportOffsets.CurrentScale));

            return convertedPos;
        }

        public static void CenterCurrentField()
        {
            var cm = BodyManager.CenterOfMass().Multi(-1.0f);
            ViewportOffsets.ViewportOffset = cm;
        }
    }
}