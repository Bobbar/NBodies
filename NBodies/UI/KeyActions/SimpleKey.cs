﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenTK;

namespace NBodies.UI.KeyActions
{
    public class SimpleKey : KeyAction
    {
        public SimpleKey(Keys key) : base(key)
        {
        }

    }
}
