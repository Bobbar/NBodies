using System;
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
using NBodies.CUDA;


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

            RenderBox.DoubleBuffered(true);
        }

        private void DisplayForm_Load(object sender, EventArgs e)
        {
            RenderVars.ScreenCenter = new PointF(this.RenderBox.Width / 2f, this.RenderBox.Height / 2f);
            RenderVars.ScaleOffset = ScaleHelpers.ScalePointExact(RenderVars.ScreenCenter);

            CUDAMain.InitGPU(2);

            Renderer.Init(RenderBox);

            MainLoop.StartLoop();
        }

        private int MouseOverID(PointF mouseLoc)
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

            return -1;
        }

        private void BodySizeTimer_Tick(object sender, EventArgs e)
        {
            BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size = BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size + 0.5;
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
                    BodyManager.Move(_selectedId, ScaleHelpers.ScalePointRelative(e.Location));
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

                    _mouseId = BodyManager.Add(ScaleHelpers.ScalePointRelative((PointF)e.Location), 0.5, ColorHelper.RandomColor());
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
                    BodyManager.FollowBodyId = -1;
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
                            BodyManager.FollowBodyId = _selectedId;
                        }
                    }
                }
            }
        }

        private void RenderBox_MouseUp(object sender, MouseEventArgs e)
        {
            _bodySizeTimer.Stop();
            _mouseDown = false;
            _selectedId = -1;
            _bodyMovin = false;

            if (_mouseId != -1)
            {
                _mouseId = -1;
                MainLoop.Resume();
            }

            if (_ctrlDown && BodyManager.FollowBodyId != -1)
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
    }
}
