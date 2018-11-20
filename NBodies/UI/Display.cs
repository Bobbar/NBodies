﻿using NBodies.Physics;
using NBodies.Rendering;
using System;
using System.Drawing;
using System.Windows.Forms;
using NBodies.UI;

namespace NBodies
{
    public partial class DisplayForm : Form
    {
        private bool _shiftDown = false;
        private bool _ctrlDown = false;
        private bool _EDown = false;
        private bool _FDown = false;



        private int _selectedUid = -1;
        private int _mouseId = -1;
        private bool _mouseRightDown = false;
        private bool _bodyMovin = false;
        private PointF _mouseMoveDownLoc = new PointF();
        private PointF _mouseLocation = new PointF();
        private Timer _bodySizeTimer = new Timer();
        private Timer _UIUpdateTimer = new Timer();
        private bool _paused = false;

        private OverlayGraphic explodeOver = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        private OverlayGraphic fpsOver = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        private OverlayGraphic flingOver = new OverlayGraphic(OverlayGraphicType.Line, new PointF(), "");
        private OverlayGraphic orbitOver = new OverlayGraphic(OverlayGraphicType.Orbit, new PointF(), "");


        private PlaybackControlForm _playbackControl;

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
            PressureScaleUpDown.Value = (decimal)Renderer.PressureScaleMax;
            AlphaUpDown.Value = Renderer.BodyAlpha;


            RenderBox.DoubleBuffered(true);
        }

        private void DisplayForm_Load(object sender, EventArgs e)
        {
            RenderVars.ScreenCenter = new PointF(this.RenderBox.Width / 2f, this.RenderBox.Height / 2f);
            RenderVars.ScaleOffset = ScaleHelpers.ScalePointExact(RenderVars.ScreenCenter);

            PhysicsProvider.InitPhysics();

            Renderer.Init(RenderBox);

            Renderer.OverLays.Add(explodeOver);
            Renderer.OverLays.Add(fpsOver);

            //var fpsKeyAction = new KeyAction(Keys.F);
            //fpsKeyAction.WheelAction = (v) =>
            //{
            //    MainLoop.TargetFPS += v;
            //    fpsOver.Value = $@"FPS Max: {MainLoop.TargetFPS}";
            //};

            //fpsKeyAction.KeyDownAction = () => 
            //{
            //    if (!Renderer.OverLays.Contains(fpsOver))
            //    {
            //        fpsOver = new OverlayGraphic(OverlayGraphicType.Text, _mouseLocation.Subtract(new PointF(10, 10)), "");
            //        fpsOver.Value = $@"FPS Max: {MainLoop.TargetFPS}";

            //        Renderer.OverLays.Add(fpsOver);
            //    }
            //};

            //fpsKeyAction.KeyUpAction = () =>
            //{
            //    fpsOver.Destroy();
            //};

            //fpsKeyAction.MouseMoveAction = (p) =>
            //{
            //    fpsOver.Location = p.Subtract(new PointF(10, 10));
            //};


            //InputHandler.AddKeyAction(fpsKeyAction);




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
            PauseButton.Checked = MainLoop.PausePhysics;

            if (PauseButton.Checked)
            {
                PauseButton.BackColor = Color.DarkRed;
            }
            else
            {
                PauseButton.BackColor = Color.DarkGreen;
            }

            FPSLabel.Text = string.Format("FPS: {0}", Math.Round(MainLoop.CurrentFPS, 2));
            FrameCountLabel.Text = string.Format("Count: {0}", MainLoop.FrameCount);
            BodyCountLabel.Text = string.Format("Bodies: {0}", BodyManager.BodyCount);
            TotalMassLabel.Text = string.Format("Tot Mass: {0}", BodyManager.TotalMass);


            var fBody = BodyManager.FollowBody();

            DensityLabel.Text = string.Format("Density: {0}", fBody.Density);
            PressureLabel.Text = string.Format("Press: {0}", fBody.Pressure);
            SpeedLabel.Text = string.Format("Agg. Speed: {0}", fBody.AggregateSpeed());


            if (_selectedUid != -1 && !MainLoop.PausePhysics)
            {
                SetSelectedInfo();
            }

            if (MainLoop.Recorder.RecordingActive)
            {
                RecordButton.BackColor = Color.DarkGreen;
                RecSizeLabel.Visible = true;
                RecSizeLabel.Text = $@"Rec Size (MB): { Math.Round((MainLoop.RecordedSize() / (float)1000000), 2) }";
            }
            else
            {
                RecordButton.BackColor = DefaultBackColor;

            }

        }

        private void SetSelectedInfo()
        {
            if (_selectedUid != -1)
            {
                var selectBody = BodyManager.BodyFromUID(_selectedUid);

                VeloXTextBox.Text = selectBody.SpeedX.ToString();
                VeloYTextBox.Text = selectBody.SpeedY.ToString();
                RadiusTextBox.Text = selectBody.Size.ToString();
                MassTextBox.Text = selectBody.Mass.ToString();
                FlagsTextBox.Text = selectBody.BlackHole.ToString();

                selectBody.PrintInfo();

            }
        }

        private void StartRecording()
        {
            if (MainLoop.Recording)
            {
                MainLoop.StopRecording();
            }

            using (var saveDialog = new SaveFileDialog())
            {
                saveDialog.Filter = "NBody Recording|*.rec";
                saveDialog.Title = "Save Recording";
                saveDialog.ShowDialog();

                if (!string.IsNullOrEmpty(saveDialog.FileName))
                {
                    MainLoop.StartRecording(saveDialog.FileName);
                }
            }
        }

        private void StartPlayback()
        {
            using (var openDialog = new OpenFileDialog())
            {
                openDialog.Filter = "NBody Recording|*.rec";
                openDialog.Title = "Load Recording";
                openDialog.ShowDialog();

                if (!string.IsNullOrEmpty(openDialog.FileName))
                {
                    var recorder = MainLoop.StartPlayback(openDialog.FileName);

                    _playbackControl?.Dispose();
                    _playbackControl = new PlaybackControlForm(recorder);
                }
            }

        }

        private void BodySizeTimer_Tick(object sender, EventArgs e)
        {
            BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size = BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size + 0.5f;
            BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Mass = BodyManager.CalcMass(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size);
        }

     


        private void DisplayForm_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.ShiftKey:

                    //   MainLoop.WaitForPause();
                    _shiftDown = true;

                    break;

                case Keys.ControlKey:

                    MainLoop.WaitForPause();
                    _ctrlDown = true;

                    break;

                case Keys.E:

                    _EDown = true;

                    //explodeOver = new OverlayGraphic(OverlayGraphicType.Text, _mouseLocation.Subtract(new PointF(10, 10)), "");
                    explodeOver.Location = _mouseLocation.Subtract(new PointF(10, 10));
                    explodeOver.Value = "Boom!";
                    explodeOver.Show();

                    Renderer.AddOverlay(explodeOver);

                    break;

                case Keys.F:

                    _FDown = true;


                    //fpsOver = new OverlayGraphic(OverlayGraphicType.Text, _mouseLocation.Subtract(new PointF(10, 10)), "");
                    fpsOver.Location = _mouseLocation.Subtract(new PointF(10, 10));
                    fpsOver.Value = $@"FPS Max: {MainLoop.TargetFPS}";
                    fpsOver.Show();

                    Renderer.AddOverlay(fpsOver);

                    break;
            }
        }

        private void DisplayForm_KeyUp(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.ShiftKey:

                    _shiftDown = false;
                    if (!_paused) MainLoop.Resume();

                    break;

                case Keys.ControlKey:

                    _ctrlDown = false;
                    if (!_paused) MainLoop.Resume();

                    break;

                case Keys.E:

                    _EDown = false;
                    explodeOver.Hide();

                    break;

                case Keys.F:

                    _FDown = false;
                    fpsOver.Hide();

                    break;

            }
        }

        private void RenderBox_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right)
            {
                if (!_mouseRightDown)
                {
                    MainLoop.WaitForPause();

                    _mouseId = BodyManager.Add(ScaleHelpers.ScalePointRelative((PointF)e.Location), 1f, ColorHelper.RandomColor());

                    //flingOver = new OverlayGraphic(OverlayGraphicType.Line, _mouseLocation, "");
                    flingOver.Location = _mouseLocation;
                    flingOver.Location2 = _mouseLocation;
                    flingOver.Show();

                    Renderer.AddOverlay(flingOver);

                    //orbitOver = new OverlayGraphic(OverlayGraphicType.Line, _mouseLocation, "");
                    orbitOver.Location = _mouseLocation;
                    orbitOver.Location2 = _mouseLocation;
                    orbitOver.Show();

                    Renderer.AddOverlay(orbitOver);

                    _mouseRightDown = true;
                    // _bodySizeTimer.Start();
                }
            }
            else if (e.Button == MouseButtons.Left)
            {
                _mouseMoveDownLoc = e.Location;

                if (_ctrlDown)
                {
                    BodyManager.FollowSelected = false;
                    BodyManager.FollowBodyUID = -1;
                }


                var mUid = MouseOverUID(e.Location);

                //if (_selectedUid == -1 && mUid != -1)
                if (mUid != -1)
                {
                    if (!_ctrlDown && _shiftDown) _bodyMovin = true;
                    _selectedUid = mUid;

                    SetSelectedInfo();

                    if (_ctrlDown)
                    {
                        BodyManager.FollowBodyUID = _selectedUid;
                    }
                }
                else if (_selectedUid != -1 && mUid == -1)
                {
                    _selectedUid = -1;
                }

                if (_EDown)
                {
                    BodyManager.InsertExplosion(ScaleHelpers.ScalePointRelative(_mouseLocation), 2500);
                }
            }
        }

        private void RenderBox_MouseUp(object sender, MouseEventArgs e)
        {
            _bodySizeTimer.Stop();

            //  _selectedUid = -1;
            _bodyMovin = false;

            if (_mouseId != -1)
            {
                _mouseId = -1;
                if (!_paused) MainLoop.Resume();
            }

            if (_ctrlDown && BodyManager.FollowBodyUID != -1)
            {
                BodyManager.FollowSelected = true;
            }

            if (_mouseRightDown)
            {
                _mouseRightDown = false;

                flingOver.Hide();
                orbitOver.Hide();
            }
        }

        private void RenderBox_MouseMove(object sender, MouseEventArgs e)
        {
            _mouseLocation = e.Location;

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
                    BodyManager.Move(BodyManager.UIDToIndex(_selectedUid), ScaleHelpers.ScalePointRelative(e.Location));
                }
                else
                {
                    var moveDiff = e.Location.Subtract(_mouseMoveDownLoc);
                    RenderVars.ViewportOffset = RenderVars.ViewportOffset.Add(ScaleHelpers.ScalePointExact(moveDiff));
                    _mouseMoveDownLoc = e.Location;
                }
            }

            if (e.Button == MouseButtons.Right)
            {
                var speedVec = _mouseLocation.Subtract(flingOver.Location);
                flingOver.Location2 = flingOver.Location.Subtract(speedVec);

                BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].SpeedX = (flingOver.Location.X - _mouseLocation.X) / 3f;
                BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].SpeedY = (flingOver.Location.Y - _mouseLocation.Y) / 3f;

                var orbitPath = BodyManager.CalcPath(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)]);//BodyManager.CalcPathCM(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)]);

                orbitOver.Location = new PointF(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].LocX, BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].LocY);
                orbitOver.OrbitPath = orbitPath;
                orbitOver.Show();
                Renderer.AddOverlay(orbitOver);
            }

            if (_EDown)
            {
                explodeOver.Location = _mouseLocation.Subtract(new PointF(10, 10));
            }

            if (_FDown)
            {
                fpsOver.Location = _mouseLocation.Subtract(new PointF(10, 10));
            }


        }

        private void RenderBox_MouseWheel(object sender, MouseEventArgs e)
        {
            var scaleChange = 0.05f * RenderVars.CurrentScale;

            if (e.Delta > 0)
            {
                if (!_FDown && !_mouseRightDown)
                    RenderVars.CurrentScale += scaleChange;

                if (_FDown)
                {
                    MainLoop.TargetFPS += 1;
                    fpsOver.Value = $@"FPS Max: {MainLoop.TargetFPS}";
                }


                if (_mouseRightDown && _mouseId != -1)
                {
                    BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size += 1.0f;
                    BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Mass = BodyManager.CalcMass(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size);
                }
            }
            else
            {
                if (!_FDown && !_mouseRightDown)
                    RenderVars.CurrentScale -= scaleChange;

                if (_FDown)
                {
                    MainLoop.TargetFPS -= 1;
                    fpsOver.Value = $@"FPS Max: {MainLoop.TargetFPS}";
                }

                if (_mouseRightDown && _mouseId != -1)
                {
                    BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size -= 1.0f;
                    BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Mass = BodyManager.CalcMass(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size);
                }
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

        private void PauseButton_Click(object sender, EventArgs e)
        {
            _paused = PauseButton.Checked;
        }

        private void saveStateToolStripMenuItem_Click(object sender, EventArgs e)
        {
            NBodies.IO.Serializer.SaveState();
        }

        private void loadStateToolStripMenuItem_Click(object sender, EventArgs e)
        {
            NBodies.IO.Serializer.LoadState();
        }

        private void reloadPreviousToolStripMenuItem_Click(object sender, EventArgs e)
        {
            NBodies.IO.Serializer.LoadPreviousState();
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

            if (!_paused) MainLoop.Resume();
        }

        private void normalToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Renderer.DisplayStyle = DisplayStyle.Normal;

            normalToolStripMenuItem.Checked = true;
            pressuresToolStripMenuItem.Checked = false;
            highContrastToolStripMenuItem1.Checked = false;
            speedsToolStripMenuItem.Checked = false;
            forcesToolStripMenuItem.Checked = false;
        }

        private void pressuresToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Renderer.DisplayStyle = DisplayStyle.Pressures;

            normalToolStripMenuItem.Checked = false;
            pressuresToolStripMenuItem.Checked = true;
            highContrastToolStripMenuItem1.Checked = false;
            speedsToolStripMenuItem.Checked = false;
            forcesToolStripMenuItem.Checked = false;
        }

        private void highContrastToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            Renderer.DisplayStyle = DisplayStyle.HighContrast;

            normalToolStripMenuItem.Checked = false;
            pressuresToolStripMenuItem.Checked = false;
            highContrastToolStripMenuItem1.Checked = true;
            speedsToolStripMenuItem.Checked = false;
            forcesToolStripMenuItem.Checked = false;
        }

        private void speedsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Renderer.DisplayStyle = DisplayStyle.Speeds;

            normalToolStripMenuItem.Checked = false;
            pressuresToolStripMenuItem.Checked = false;
            highContrastToolStripMenuItem1.Checked = false;
            speedsToolStripMenuItem.Checked = true;
            forcesToolStripMenuItem.Checked = false;
        }

        private void forcesToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Renderer.DisplayStyle = DisplayStyle.Forces;

            normalToolStripMenuItem.Checked = false;
            pressuresToolStripMenuItem.Checked = false;
            highContrastToolStripMenuItem1.Checked = false;
            speedsToolStripMenuItem.Checked = false;
            forcesToolStripMenuItem.Checked = true;
        }

        private void UpdateButton_Click(object sender, EventArgs e)
        {
            if (_selectedUid != -1)
            {
                BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].SpeedX = Convert.ToSingle(VeloXTextBox.Text.Trim());
                BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].SpeedY = Convert.ToSingle(VeloYTextBox.Text.Trim());
                BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].Size = Convert.ToSingle(RadiusTextBox.Text.Trim());
                BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].Mass = Convert.ToSingle(MassTextBox.Text.Trim());
                BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].BlackHole = Convert.ToInt32(FlagsTextBox.Text.Trim());
            }
        }

        private void PressureScaleUpDown_ValueChanged(object sender, EventArgs e)
        {
            Renderer.PressureScaleMax = (float)PressureScaleUpDown.Value;
        }

        private void CenterOnMassButton_Click(object sender, EventArgs e)
        {
            var cm = BodyManager.CenterOfMass().Multi(-1.0f);

            RenderVars.ViewportOffset = cm;
        }

        private void ScreenShotButton_Click(object sender, EventArgs e)
        {
            // BodyManager.TotEnergy();
            // BodyManager.CalcPath(BodyManager.FollowBody());
            //   BodyManager.SetVelo(40, 0);

            BodyManager.ShiftPos(20000, 0);
        }

        private void AlphaUpDown_ValueChanged(object sender, EventArgs e)
        {
            Renderer.BodyAlpha = (int)AlphaUpDown.Value;
        }

        private void showFollowBodyForce_CheckedChanged(object sender, EventArgs e)
        {
            Renderer.ShowForce = showFollowBodyForce.Checked;
        }

        private void showPredictOrbit_CheckedChanged(object sender, EventArgs e)
        {
            Renderer.ShowPath = showPredictOrbit.Checked;
        }

        private void DisplayForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            MainLoop.Stop();
        }



        private void LoadRecordingButton_Click(object sender, EventArgs e)
        {
            StartPlayback();
        }

        private void RecordButton_Click(object sender, EventArgs e)
        {
            if (MainLoop.Recording)
            {
                MainLoop.StopRecording();
            }
            else
            {
                StartRecording();
            }
        }

        private void drawToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            MainLoop.DrawBodies = drawToolStripMenuItem.Checked;
        }
    }
}