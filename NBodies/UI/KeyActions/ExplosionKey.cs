﻿using NBodies.Helpers;
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

        public ExplosionKey()
        {
            AddKey(Keys.E);
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

        public override void DoMouseDown(MouseButtons button, PointF mouseLoc)
        {
            //if (KeyDownStates[Keys.E])
            //    BodyManager.InsertExplosion(ViewportHelpers.ScreenPointToField(mouseLoc), 2500);
        }

        public override void DoMouseDown(MouseButtons button, Vector3 loc)
        {
            if (KeyDownStates[Keys.E])
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
        }

        public override void DoMouseMove(PointF mouseLoc)
        {
            //throw new NotImplementedException();
        }

        public override void DoMouseUp(MouseButtons button, PointF mouseLoc)
        {
            MainLoop.Shooting = false;
        }

        public override void DoWheelAction(int wheelValue)
        {
            if (KeyDownStates[Keys.E])
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
}