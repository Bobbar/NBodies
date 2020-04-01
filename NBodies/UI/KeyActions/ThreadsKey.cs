﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;
using OpenTK;

namespace NBodies.UI.KeyActions
{
    public class ThreadsKey : KeyAction
    {
        public ThreadsKey(Keys key) : base(key)
        {
            Overlay = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        }

        public override void DoKeyDown()
        {
            Overlay.Value = "Threads Per Block: " + (int)Math.Pow(2, MainLoop.ThreadsPerBlock);
            Overlay.Show();
        }

        public override void DoKeyUp()
        {
            Overlay.Hide();
        }

        public override void DoWheelAction(int wheelValue)
        {
            MainLoop.ThreadsPerBlock += wheelValue;
            Overlay.Value = "Threads Per Block: " + (int)Math.Pow(2, MainLoop.ThreadsPerBlock);
        }
    }
}
