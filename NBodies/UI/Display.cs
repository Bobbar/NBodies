using NBodies.Physics;
using NBodies.Rendering;
using NBodies.Extensions;
using System;
using System.Drawing;
using System.Windows.Forms;

namespace NBodies
{
    public partial class DisplayForm : Form
    {
        private bool _shiftDown = false;
        private bool _ctrlDown = false;
        private bool _EDown = false;
        private bool _FDown = false;
        private bool _DDown = false;
        private bool _CDown = false;

        private bool _hideToolbar = false;
        private float _ogToolbarHeight;

        private int _selectedUid = -1;
        private int _mouseId = -1;
        private bool _mouseRightDown = false;
        private bool _bodyMovin = false;
        private PointF _mouseMoveDownLoc = new PointF();
        private PointF _mouseLocation = new PointF();
        private Point _flingPrevScreenPos = new Point();
        private PointF _flingStartPos = new PointF();
        private PointF _flingVirtMousePos = new PointF();

        private Timer _UIUpdateTimer = new Timer();

        private bool _paused = false;

        private OverlayGraphic explodeOver = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        private OverlayGraphic fpsOver = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        private OverlayGraphic flingOver = new OverlayGraphic(OverlayGraphicType.Line, new PointF(), "");
        private OverlayGraphic orbitOver = new OverlayGraphic(OverlayGraphicType.Orbit, new PointF(), "");
        private OverlayGraphic distLine = new OverlayGraphic(OverlayGraphicType.Line, new PointF(), "");
        private OverlayGraphic distOver = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");
        private OverlayGraphic cellSizeOver = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");

        private PlaybackControlForm _playbackControl;
        private bool _useD2D = true;

        public DisplayForm()
        {
            InitializeComponent();

            _UIUpdateTimer.Interval = 100;
            _UIUpdateTimer.Tick += _UIUpdateTimer_Tick;
            _UIUpdateTimer.Start();

            RenderBox.MouseWheel += RenderBox_MouseWheel;

            TimeStepUpDown.Value = (decimal)MainLoop.TimeStep;
            PressureScaleUpDown.Value = (decimal)RenderBase.StyleScaleMax;
            AlphaUpDown.Value = RenderBase.BodyAlpha;

            RenderBox.DoubleBuffered(true);
        }

        private void DisplayForm_Load(object sender, EventArgs e)
        {
            RenderVars.ScreenCenter = new PointF(this.RenderBox.Width / 2f, this.RenderBox.Height / 2f);
            RenderVars.ScaleOffset = ScaleHelpers.FieldPointToScreenUnscaled(RenderVars.ScreenCenter);

            PhysicsProvider.InitPhysics();

            MainLoop.Renderer = new D2DRenderer(RenderBox);

            RenderBase.OverLays.Add(explodeOver);
            RenderBase.OverLays.Add(fpsOver);

            RenderBase.OverLays.Add(distLine);
            RenderBase.OverLays.Add(distOver);
            RenderBase.OverLays.Add(cellSizeOver);

            MainLoop.StartLoop();
        }


        private void SwitchRenderer()
        {
            MainLoop.DrawBodies = false;
            MainLoop.WaitForPause();
            MainLoop.Renderer.Destroy();

            _useD2D = !_useD2D;

            if (_useD2D)
            {
                MainLoop.Renderer = new D2DRenderer(RenderBox);
            }
            else
            {
                MainLoop.Renderer = new GDIRenderer(RenderBox);
            }

            MainLoop.DrawBodies = true;
        }

        private int MouseOverUID(PointF mouseLoc)
        {
            try
            {
                for (int i = 0; i < BodyManager.Bodies.Length; i++)
                {
                    var body = BodyManager.Bodies[i];
                    var dist = Math.Sqrt(Math.Pow(ScaleHelpers.ScreenPointToField(mouseLoc).X - body.LocX, 2) + Math.Pow(ScaleHelpers.ScreenPointToField(mouseLoc).Y - body.LocY, 2));

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
            ScaleLabel.Text = string.Format("Scale: {0}", Math.Round(RenderVars.CurrentScale, 2));

            RendererLabel.Text = $@"Renderer: { MainLoop.Renderer.ToString() }";

            if (BodyManager.FollowSelected)
            {
                DensityLabel.Visible = true;
                PressureLabel.Visible = true;
                SpeedLabel.Visible = true;
            }
            else
            {
                DensityLabel.Visible = false;
                PressureLabel.Visible = false;
                SpeedLabel.Visible = false;
            }

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
                RecSizeLabel.Visible = false;
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

                DensityLabel.Text = string.Format("Density: {0}", selectBody.Density);
                PressureLabel.Text = string.Format("Press: {0}", selectBody.Pressure);
                SpeedLabel.Text = string.Format("Agg. Speed: {0}", selectBody.AggregateSpeed());

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

                    RenderBase.AddOverlay(explodeOver);

                    break;

                case Keys.F:

                    _FDown = true;

                    //fpsOver = new OverlayGraphic(OverlayGraphicType.Text, _mouseLocation.Subtract(new PointF(10, 10)), "");
                    fpsOver.Location = _mouseLocation.Subtract(new PointF(10, 10));
                    fpsOver.Value = $@"FPS Max: {MainLoop.TargetFPS}";
                    fpsOver.Show();

                    RenderBase.AddOverlay(fpsOver);

                    break;

                case Keys.B:

                    if (_selectedUid != -1)
                    {
                        BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].BlackHole = 2;
                    }

                    break;

                case Keys.D:

                    if (!_DDown)
                    {
                        distLine.Location = _mouseLocation;
                        distOver.Location = _mouseLocation.Subtract(new PointF(10, 10));
                        distOver.Value = "0.0";
                        distLine.Show();
                        distOver.Show();
                    }

                    _DDown = true;

                    break;

                case Keys.C:

                    _CDown = true;

                    cellSizeOver.Location = _mouseLocation.Subtract(new PointF(10, 20));
                    cellSizeOver.Value = "Cell Size: " + Math.Pow(2, MainLoop.CellSizeExp).ToString();
                    cellSizeOver.Show();
                    break;

                case Keys.F11:

                    if (!_hideToolbar)
                    {
                        _ogToolbarHeight = RootLayoutTable.RowStyles[0].Height;
                        RootLayoutTable.RowStyles[0].Height = 0;
                        this.FormBorderStyle = FormBorderStyle.None;
                        _hideToolbar = true;
                    }
                    else
                    {
                        RootLayoutTable.RowStyles[0].Height = _ogToolbarHeight;
                        this.FormBorderStyle = FormBorderStyle.Sizable;
                        _hideToolbar = false;
                    }

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

                case Keys.D:

                    _DDown = false;
                    distLine.Hide();
                    distOver.Hide();

                    break;

                case Keys.C:

                    _CDown = false;
                    cellSizeOver.Hide();

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

                    var mUid = MouseOverUID(e.Location);

                    if (mUid != -1)
                    {
                        _mouseId = mUid;
                    }
                    else
                    {
                        _mouseId = BodyManager.Add(ScaleHelpers.ScreenPointToField(e.Location), 1f, ColorHelper.RandomColor());
                    }

                    var bodyPos = ScaleHelpers.FieldPointToScreen(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Position());

                    flingOver.Location = bodyPos;
                    flingOver.Location2 = bodyPos;
                    flingOver.Show();

                    RenderBase.AddOverlay(flingOver);

                    orbitOver.Location = bodyPos;
                    orbitOver.Location2 = bodyPos;
                    orbitOver.Show();

                    RenderBase.AddOverlay(orbitOver);

                    _mouseRightDown = true;
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
                    BodyManager.InsertExplosion(ScaleHelpers.ScreenPointToField(_mouseLocation), 2500);
                }
            }
        }

        private void RenderBox_MouseUp(object sender, MouseEventArgs e)
        {
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

            _flingStartPos = new PointF();
            _flingPrevScreenPos = new Point();
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
                    BodyManager.Move(BodyManager.UIDToIndex(_selectedUid), ScaleHelpers.ScreenPointToField(e.Location));
                }
                else
                {
                    var moveDiff = e.Location.Subtract(_mouseMoveDownLoc);
                    RenderVars.ViewportOffset = RenderVars.ViewportOffset.Add(ScaleHelpers.FieldPointToScreenUnscaled(moveDiff));
                    _mouseMoveDownLoc = e.Location;
                }
            }

            if (e.Button == MouseButtons.Right)
            {
                // This logic locks the mouse pointer to the body position and calculates a 'virtual' mouse location.
                // This is done to allow infinite fling deflection without the mouse stopping at the edge of a screen.

                // If the mouse has moved from its previous position.
                if (_flingPrevScreenPos != Cursor.Position)
                {
                    // Calculate the new virtual position from the previous position.
                    _flingVirtMousePos = _flingVirtMousePos.Add(_flingPrevScreenPos.Subtract(Cursor.Position));

                    // Record the initial position at the start of a fling.
                    if (_flingStartPos == new PointF())
                    {
                        _flingStartPos = _flingVirtMousePos;
                    }

                    // Calculate the amount of deflection from the start position.
                    var deflection = _flingStartPos.Subtract(_flingVirtMousePos);

                    // Update the fling overlay location to visualize the resultant vector.
                    // VectorPos2 = VectorPos1 - deflection
                    flingOver.Location2 = flingOver.Location.Subtract(deflection);

                    // Flip and shorten the vector and apply it to the body speed.
                    BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].SpeedX = -deflection.X / 3f;
                    BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].SpeedY = -deflection.Y / 3f;

                    // Calculate the new orbital path.
                    var orbitPath = BodyManager.CalcPathCircle(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)]);

                    // Update the orbit overlay.
                    orbitOver.Location = new PointF(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].LocX, BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].LocY);
                    orbitOver.OrbitPath = orbitPath;
                    orbitOver.Show();
                    RenderBase.AddOverlay(orbitOver);
                }

                // Calculate the true screen position from the body location.
                var clientPosition = ScaleHelpers.FieldPointToScreen(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Position());
                var screenPosition = RenderBox.PointToScreen(clientPosition.ToPoint());

                // Lock the cursor in place above the body.
                Cursor.Position = screenPosition;
                _flingPrevScreenPos = screenPosition;

            }

            if (_EDown)
            {
                explodeOver.Location = _mouseLocation.Subtract(new PointF(10, 10));
            }

            if (_FDown)
            {
                fpsOver.Location = _mouseLocation.Subtract(new PointF(10, 10));
            }

            if (_DDown)
            {
                distLine.Location2 = _mouseLocation;
                distOver.Location = _mouseLocation.Add(new PointF(40, 10));

                var loc1 = ScaleHelpers.ScreenPointToField(distLine.Location);
                var loc2 = ScaleHelpers.ScreenPointToField(distLine.Location2);

                distOver.Value = loc1.DistanceSqrt(loc2).ToString();
            }

            if (_CDown)
            {
                cellSizeOver.Location = _mouseLocation.Subtract(new PointF(10, 20));
            }
        }

        private void RenderBox_MouseWheel(object sender, MouseEventArgs e)
        {
            var scaleChange = 0.05f * RenderVars.CurrentScale;

            if (e.Delta > 0)
            {
                if (_CDown)
                {
                    MainLoop.CellSizeExp += 1;
                    cellSizeOver.Value = "Cell Size: " + Math.Pow(2, MainLoop.CellSizeExp).ToString();
                }

                if (!_FDown && !_CDown && !_mouseRightDown)
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
                if (_CDown)
                {
                    MainLoop.CellSizeExp -= 1;
                    cellSizeOver.Value = "Cell Size: " + Math.Pow(2, MainLoop.CellSizeExp).ToString();
                }

                if (!_FDown && !_CDown && !_mouseRightDown)
                    RenderVars.CurrentScale -= scaleChange;

                if (_FDown)
                {
                    MainLoop.TargetFPS -= 1;
                    fpsOver.Value = $@"FPS Max: {MainLoop.TargetFPS}";
                }

                if (_mouseRightDown && _mouseId != -1)
                {
                    if (BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size - 1.0f > 0.5f)
                    {
                        BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size -= 1.0f;
                        BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Mass = BodyManager.CalcMass(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size);
                    }
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
            RenderBase.Trails = TrailsCheckBox.Checked;
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
            RenderBase.AAEnabled = antiAliasingToolStripMenuItem.Checked;
        }

        private void clipToViewportToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            RenderBase.ClipView = clipToViewportToolStripMenuItem.Checked;
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
            RenderBase.DisplayStyle = DisplayStyle.Normal;

            normalToolStripMenuItem.Checked = true;
            pressuresToolStripMenuItem.Checked = false;
            highContrastToolStripMenuItem1.Checked = false;
            speedsToolStripMenuItem.Checked = false;
            forcesToolStripMenuItem.Checked = false;
        }

        private void pressuresToolStripMenuItem_Click(object sender, EventArgs e)
        {
            RenderBase.DisplayStyle = DisplayStyle.Pressures;

            normalToolStripMenuItem.Checked = false;
            pressuresToolStripMenuItem.Checked = true;
            highContrastToolStripMenuItem1.Checked = false;
            speedsToolStripMenuItem.Checked = false;
            forcesToolStripMenuItem.Checked = false;
        }

        private void highContrastToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            RenderBase.DisplayStyle = DisplayStyle.HighContrast;

            normalToolStripMenuItem.Checked = false;
            pressuresToolStripMenuItem.Checked = false;
            highContrastToolStripMenuItem1.Checked = true;
            speedsToolStripMenuItem.Checked = false;
            forcesToolStripMenuItem.Checked = false;
        }

        private void speedsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            RenderBase.DisplayStyle = DisplayStyle.Speeds;

            normalToolStripMenuItem.Checked = false;
            pressuresToolStripMenuItem.Checked = false;
            highContrastToolStripMenuItem1.Checked = false;
            speedsToolStripMenuItem.Checked = true;
            forcesToolStripMenuItem.Checked = false;
        }

        private void forcesToolStripMenuItem_Click(object sender, EventArgs e)
        {
            RenderBase.DisplayStyle = DisplayStyle.Forces;

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
            RenderBase.StyleScaleMax = (float)PressureScaleUpDown.Value;
        }

        private void CenterOnMassButton_Click(object sender, EventArgs e)
        {
            var cm = BodyManager.CenterOfMass().Multi(-1.0f);

            RenderVars.ViewportOffset = cm;
        }

        private void ToggleRendererButton_Click(object sender, EventArgs e)
        {
            SwitchRenderer();
        }

        private void AlphaUpDown_ValueChanged(object sender, EventArgs e)
        {
            RenderBase.BodyAlpha = (int)AlphaUpDown.Value;
        }

        private void showFollowBodyForce_CheckedChanged(object sender, EventArgs e)
        {
            RenderBase.ShowForce = showFollowBodyForce.Checked;
        }

        private void showPredictOrbit_CheckedChanged(object sender, EventArgs e)
        {
            RenderBase.ShowPath = showPredictOrbit.Checked;
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

        private void rocheLimitToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            MainLoop.RocheLimit = rocheLimitToolStripMenuItem.Checked;
        }

        private void leapfrogIntegratorToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            MainLoop.LeapFrog = leapfrogIntegratorToolStripMenuItem.Checked;
        }

        private void showMeshToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            RenderBase.ShowMesh = showMeshToolStripMenuItem.Checked;
        }

        private void allForceVectorsToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            RenderBase.ShowAllForce = allForceVectorsToolStripMenuItem.Checked;
        }
    }
}