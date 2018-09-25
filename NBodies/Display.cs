﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;
using NBodies.Structures;


namespace NBodies
{
    public partial class DisplayForm : Form
    {
        private bool _shiftDown = false;
        private bool _ctrlDown = false;
        private int _selectedId = -1;
        private int _mouseId = -1;
        private bool _mouseDown = false;
        private bool _bodyMovin = false;
        private PointF _mouseMoveDown = new PointF();

        private Timer _bodySizeTimer = new Timer();

        public DisplayForm()
        {
            InitializeComponent();

            _bodySizeTimer.Interval = 10;
            _bodySizeTimer.Tick += BodySizeTimer_Tick;
            RenderBox.MouseWheel += RenderBox_MouseWheel;
        }

     

        private void DisplayForm_Load(object sender, EventArgs e)
        {
            RenderVars.ScreenCenter = new PointF(this.RenderBox.Width / 2f, this.RenderBox.Height / 2f);
            RenderVars.ScaleOffset = ScaleHelpers.ScaleMousePosExact(RenderVars.ScreenCenter);
        }

        private int MouseOverID(PointF mouseLoc)
        {
            for (int i = 0; i < BodyManager.Bodies.Length; i++)
            {
                var body = BodyManager.Bodies[i];
                var dist = Math.Sqrt(Math.Pow(ScaleHelpers.ScaleMousePosRelative(mouseLoc).X - body.LocX, 2) + Math.Pow(ScaleHelpers.ScaleMousePosRelative(mouseLoc).Y - body.LocY, 2));

                if (dist < body.Size * 0.5f)
                {
                    return i;
                }
            }

            return -1;
        }

        private void BodySizeTimer_Tick(object sender, EventArgs e)
        {
            BodyManager.Bodies[_mouseId].Size = BodyManager.Bodies[_mouseId].Size + 0.5;
        }

        private void RenderBox_MouseWheel(object sender, MouseEventArgs e)
        {
            var scaleChange = 0.05f * RenderVars.CurrentScale;

            if (e.Delta > 0)
            {
                RenderVars.CurrentScale += scaleChange;
            }
            else
            {
                RenderVars.CurrentScale -= scaleChange;
            }

        }

        private void RenderBox_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                if (_selectedId == -1 && _shiftDown)
                {
                    var mId = MouseOverID(e.Location);
                    if (mId != -1)
                    {
                        _bodyMovin = true;
                        _selectedId = mId;
                    }
                }

                if (_bodyMovin)
                {
                    BodyManager.Move(_selectedId, ScaleHelpers.ScaleMousePosRelative(e.Location));
                }
                else
                {
                    var moveDiff = e.Location.Subtract(_mouseMoveDown);
                    RenderVars.ViewportOffset = RenderVars.ViewportOffset.Add(ScaleHelpers.ScaleMousePosExact(moveDiff));
                    _mouseMoveDown = e.Location;
                }
            }
        }

        private void DisplayForm_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.ShiftKey:
                    _shiftDown = true;
                    break;

                case Keys.Control:
                    _ctrlDown = true;
                    break;
            }
        }

        private void DisplayForm_KeyUp(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.ShiftKey:
                    _shiftDown = false;
                    break;

                case Keys.Control:
                    _ctrlDown = false;
                    break;
            }
        }

        private void RenderBox_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right)
            {
                if (!_mouseDown)
                {
                    _mouseId = BodyManager.Add(ScaleHelpers.ScaleMousePosRelative((PointF)e.Location), 0.5, ColorHelper.RandomColor());
                    _mouseDown = true;
                    _bodySizeTimer.Start();
                }
            }
            else if (e.Button == MouseButtons.Left)
            {
                _mouseMoveDown = e.Location;

                if (_ctrlDown)
                {
                    BodyManager.FollowSelected = false;
                    BodyManager.FollowBodyIndex = -1;
                }

                if (_selectedId == -1)
                {
                    var mId = MouseOverID(e.Location);
                    if (mId != -1)
                    {
                        if (!_ctrlDown && _shiftDown) _bodyMovin = true;
                        _selectedId = mId;

                        if (_ctrlDown)
                        {
                            BodyManager.FollowBodyIndex = _selectedId;
                        }
                    }
                }
            }
        }

        private void RenderBox_MouseUp(object sender, MouseEventArgs e)
        {
            _bodySizeTimer.Stop();

            _selectedId = -1;
            _bodyMovin = false;

            if (_ctrlDown && BodyManager.FollowBodyIndex != -1)
            {
                BodyManager.FollowSelected = true;
            }

        }


    }
}
