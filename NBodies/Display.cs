using NBodies.Physics;
using NBodies.Rendering;
using System;
using System.Drawing;
using System.Windows.Forms;

namespace NBodies
{
    public partial class DisplayForm : Form
    {
        private bool _shiftDown = false;
        private bool _ctrlDown = false;
        private int _selectedUid = -1;
        private int _mouseId = -1;
        private bool _mouseDown = false;
        private bool _bodyMovin = false;
        private PointF _mouseMoveDown = new PointF();
        private Timer _bodySizeTimer = new Timer();
        private Timer _UIUpdateTimer = new Timer();

        public DisplayForm()
        {
            InitializeComponent();

            _bodySizeTimer.Interval = 10;
            _bodySizeTimer.Tick += BodySizeTimer_Tick;

            _UIUpdateTimer.Interval = 100;
            _UIUpdateTimer.Tick += _UIUpdateTimer_Tick;
            _UIUpdateTimer.Start();

            RenderBox.MouseWheel += RenderBox_MouseWheel;

            TimeStepUpDown.Value = (decimal)MainLoop.TimeStep;

            RenderBox.DoubleBuffered(true);
        }

        private void DisplayForm_Load(object sender, EventArgs e)
        {
            RenderVars.ScreenCenter = new PointF(this.RenderBox.Width / 2f, this.RenderBox.Height / 2f);
            RenderVars.ScaleOffset = ScaleHelpers.ScalePointExact(RenderVars.ScreenCenter);

            PhysicsProvider.InitPhysics();

            Renderer.Init(RenderBox);

            MainLoop.StartLoop();
        }

        private int MouseOverUID(PointF mouseLoc)
        {
            try
            {
                for (int i = 0; i < BodyManager.Bodies.Length; i++)
                {
                    var body = BodyManager.Bodies[i];
                    var dist = Math.Sqrt(Math.Pow(ScaleHelpers.ScalePointRelative(mouseLoc).X - body.LocX, 2) + Math.Pow(ScaleHelpers.ScalePointRelative(mouseLoc).Y - body.LocY, 2));

                    if (dist < body.Size * 0.5f)
                    {
                        return body.UID;
                    }
                }
            }
            catch (IndexOutOfRangeException)
            {
                // Fail silently
            }

            return -1;
        }

        private void _UIUpdateTimer_Tick(object sender, EventArgs e)
        {
            FPSLabel.Text = string.Format("FPS: {0}", Math.Round(MainLoop.CurrentFPS, 2));
            FrameCountLabel.Text = string.Format("Count: {0}", MainLoop.FrameCount);
            BodyCountLabel.Text = string.Format("Bodies: {0}", BodyManager.BodyCount);
            TotalMassLabel.Text = string.Format("Tot Mass: {0}", BodyManager.TotalMass);

            DensityLabel.Text = string.Format("Density: {0}", BodyManager.FollowBody().Density);
            PressureLabel.Text = string.Format("Press: {0}", BodyManager.FollowBody().Pressure);
        }

        private void BodySizeTimer_Tick(object sender, EventArgs e)
        {
            BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size = BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size + 0.5f;
            BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Mass = BodyManager.CalcMass(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size);
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
                if (_selectedUid == -1 && _shiftDown)
                {
                    var mId = MouseOverUID(e.Location);
                    if (mId != -1)
                    {
                        _bodyMovin = true;
                        _selectedUid = mId;
                    }
                }

                if (_bodyMovin)
                {
                    BodyManager.Move(_selectedUid, ScaleHelpers.ScalePointRelative(e.Location));
                }
                else
                {
                    var moveDiff = e.Location.Subtract(_mouseMoveDown);
                    RenderVars.ViewportOffset = RenderVars.ViewportOffset.Add(ScaleHelpers.ScalePointExact(moveDiff));
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

                case Keys.ControlKey:

                    MainLoop.WaitForPause();
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

                case Keys.ControlKey:
                    _ctrlDown = false;
                    break;
            }

            MainLoop.Resume();
        }

        private void RenderBox_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right)
            {
                if (!_mouseDown)
                {
                    MainLoop.WaitForPause();

                    _mouseId = BodyManager.Add(ScaleHelpers.ScalePointRelative((PointF)e.Location), 0.5f, ColorHelper.RandomColor());
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
                    BodyManager.FollowBodyUID = -1;
                }

                if (_selectedUid == -1)
                {
                    var mUid = MouseOverUID(e.Location);
                    if (mUid != -1)
                    {
                        if (!_ctrlDown && _shiftDown) _bodyMovin = true;
                        _selectedUid = mUid;

                        if (_ctrlDown)
                        {
                            BodyManager.FollowBodyUID = _selectedUid;
                        }
                    }
                }
            }
        }

        private void RenderBox_MouseUp(object sender, MouseEventArgs e)
        {
            _bodySizeTimer.Stop();
            _mouseDown = false;
            _selectedUid = -1;
            _bodyMovin = false;

            if (_mouseId != -1)
            {
                _mouseId = -1;
                MainLoop.Resume();
            }

            if (_ctrlDown && BodyManager.FollowBodyUID != -1)
            {
                BodyManager.FollowSelected = true;
            }
        }

        private void RenderBox_Resize(object sender, EventArgs e)
        {
            RenderVars.ScreenCenter = new PointF(this.RenderBox.Width * 0.5f, this.RenderBox.Height * 0.5f);
        }

        private void AddBodiesButton_Click(object sender, EventArgs e)
        {
            new AddBodiesForm().Show();
        }

        private void TrailsCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            Renderer.Trails = TrailsCheckBox.Checked;
        }

        private void PauseButton_CheckedChanged(object sender, EventArgs e)
        {
            MainLoop.PausePhysics = PauseButton.Checked;
        }

        private void saveStateToolStripMenuItem_Click(object sender, EventArgs e)
        {
            NBodies.IO.Serializer.SaveState();
        }

        private void loadStateToolStripMenuItem_Click(object sender, EventArgs e)
        {
            NBodies.IO.Serializer.LoadState();
        }

        private void antiAliasingToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            Renderer.AntiAliasing = antiAliasingToolStripMenuItem.Checked;
        }

        private void clipToViewportToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            Renderer.ClipView = clipToViewportToolStripMenuItem.Checked;
        }

        private void TimeStepUpDown_ValueChanged(object sender, EventArgs e)
        {
            MainLoop.TimeStep = (float)TimeStepUpDown.Value;
        }

        private void RemoveAllButton_Click(object sender, EventArgs e)
        {
            MainLoop.WaitForPause();

            BodyManager.ClearBodies();

            MainLoop.Resume();
        }

        private void normalToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Renderer.DisplayStyle = DisplayStyle.Normal;

            normalToolStripMenuItem.Checked = true;
            pressuresToolStripMenuItem.Checked = false;
            highContrastToolStripMenuItem1.Checked = false;
        }

        private void pressuresToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Renderer.DisplayStyle = DisplayStyle.Pressures;

            normalToolStripMenuItem.Checked = false;
            pressuresToolStripMenuItem.Checked = true;
            highContrastToolStripMenuItem1.Checked = false;
        }

        private void highContrastToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            Renderer.DisplayStyle = DisplayStyle.HighContrast;

            normalToolStripMenuItem.Checked = false;
            pressuresToolStripMenuItem.Checked = false;
            highContrastToolStripMenuItem1.Checked = true;
        }

        private void clipToViewportToolStripMenuItem_Click(object sender, EventArgs e)
        {
        }
    }
}