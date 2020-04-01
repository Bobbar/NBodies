using NBodies.Helpers;
using NBodies.Physics;
using NBodies.Rendering;
using System.Drawing;
using System.Windows.Forms;
using OpenTK;
using System;

namespace NBodies.UI.KeyActions
{
    internal class ExplosionKey : KeyAction
    {
        private bool _explode = true;

        public ExplosionKey(Keys key) : base(key)
        {
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            if (_explode)
                Overlay.Value = "Boom!";
            else
                Overlay.Value = "Shoot!";

            Overlay.Show();
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
            MainLoop.Shooting = false;
        }

        public override void DoMouseDown(MouseButtons button, Vector3 loc)
        {
            if (_explode)
            {
                BodyManager.InsertExplosion(loc, 2500);
            }
            else
            {
                MainLoop.Shooting = true;
            }
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
            MainLoop.Shooting = false;
        }

        public override void DoWheelAction(int wheelValue)
        {
            _explode = !_explode;

            if (_explode)
                Overlay.Value = "Boom!";
            else
                Overlay.Value = "Shoot!";

            Overlay.Show();
        }
    }
}