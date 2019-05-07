using NBodies.Extensions;
using NBodies.Physics;
using NBodies.Rendering;
using NBodies.UI.KeyActions;
using NBodies.Helpers;
using System;
using System.Drawing;
using System.Windows.Forms;

namespace NBodies.UI
{
    public partial class DisplayForm : Form
    {
        private AddBodiesForm _addFormInstance = new AddBodiesForm();
        private PlaybackControlForm _playbackControl;

        private bool _shiftDown = false;
        private bool _ctrlDown = false;
        private bool _mouseRightDown = false;
        private bool _hideToolbar = false;
        private float _ogToolbarHeight;

        private int _selectedUid = -1;
        private int _mouseId = -1;
        private bool _bodyMovin = false;
        private PointF _mouseMoveDownLoc = new PointF();
        private PointF _mouseLocation = new PointF();
        private Point _flingPrevScreenPos = new Point();
        private PointF _flingStartPos = new PointF();
        private PointF _flingVirtMousePos = new PointF();

        private Timer _UIUpdateTimer = new Timer();

        private OverlayGraphic _flingOver = new OverlayGraphic(OverlayGraphicType.Line, new PointF(), "");
        private OverlayGraphic _orbitOver = new OverlayGraphic(OverlayGraphicType.Orbit, new PointF(), "");
        private OverlayGraphic _distLine = new OverlayGraphic(OverlayGraphicType.Line, new PointF(), "");
        private OverlayGraphic _distOver = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");

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
            ViewportOffsets.ScreenCenter = new PointF(this.RenderBox.Width / 2f, this.RenderBox.Height / 2f);
            ViewportOffsets.ScaleOffset = ViewportHelpers.FieldPointToScreenNoOffset(ViewportOffsets.ScreenCenter);
            MainLoop.MaxThreadsPerBlock = Program.ThreadsPerBlockArgument;
            PhysicsProvider.InitPhysics();

            MainLoop.Renderer = new D2DRenderer(RenderBox);

            RenderBase.OverLays.Add(_distLine);
            RenderBase.OverLays.Add(_distOver);

            InputHandler.AddKeyAction(new FPSKey());
            InputHandler.AddKeyAction(new ExplosionKey());
            InputHandler.AddKeyAction(new CellSizeKey());
            InputHandler.AddKeyAction(new DisplayStyleKey());
            InputHandler.AddKeyAction(new AlphaKey());
            InputHandler.AddKeyAction(new SimpleKey(Keys.D));
            InputHandler.AddKeyAction(new TimeStepKey());
            InputHandler.AddKeyAction(new RewindKey());
            InputHandler.AddKeyAction(new LevelKey());
            InputHandler.AddKeyAction(new ThreadsKey());
            InputHandler.AddKeyAction(new ViscosityKey());


            SetDisplayOptionTags();

            MainLoop.StartLoop();
        }

        private void SwitchRenderer()
        {
            MainLoop.DrawBodies = false;
            MainLoop.WaitForPause();
            MainLoop.Stop();
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
            MainLoop.StartLoop();
            MainLoop.ResumePhysics();
        }

        private int MouseOverUID(PointF mouseLoc)
        {
            try
            {
                for (int i = 0; i < BodyManager.Bodies.Length; i++)
                {
                    var body = BodyManager.Bodies[i];
                    var dist = Math.Sqrt(Math.Pow(ViewportHelpers.ScreenPointToField(mouseLoc).X - body.PosX, 2) + Math.Pow(ViewportHelpers.ScreenPointToField(mouseLoc).Y - body.PosY, 2));

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
            TotalMassLabel.Text = string.Format("Tot Mass: {0}", Math.Round(BodyManager.TotalMass,2));
            ScaleLabel.Text = string.Format("Scale: {0}", Math.Round(ViewportOffsets.CurrentScale, 2));
            AlphaUpDown.Value = RenderBase.BodyAlpha;
            TimeStepUpDown.Value = (decimal)MainLoop.TimeStep;
            SetDisplayStyle(RenderBase.DisplayStyle);

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

                VeloXTextBox.Text = selectBody.VeloX.ToString();
                VeloYTextBox.Text = selectBody.VeloY.ToString();
                RadiusTextBox.Text = selectBody.Size.ToString();
                MassTextBox.Text = selectBody.Mass.ToString();
                FlagsTextBox.Text = selectBody.Flag.ToString();

                DensityLabel.Text = string.Format("Density: {0}", selectBody.Density);
                PressureLabel.Text = string.Format("Press: {0}", selectBody.Pressure);
                SpeedLabel.Text = string.Format("Agg. Speed: {0}", selectBody.AggregateSpeed());

                // selectBody.PrintInfo();
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

        private void SetDisplayOptionTags()
        {
            normalToolStripMenuItem.Tag = DisplayStyle.Normal;
            pressuresToolStripMenuItem.Tag = DisplayStyle.Pressures;
            highContrastToolStripMenuItem.Tag = DisplayStyle.HighContrast;
            speedsToolStripMenuItem.Tag = DisplayStyle.Speeds;
            indexToolStripMenuItem.Tag = DisplayStyle.Index;
            forcesToolStripMenuItem.Tag = DisplayStyle.Forces;
        }

        private void SetDisplayStyle(DisplayStyle style)
        {
            RenderBase.DisplayStyle = style;

            foreach (ToolStripMenuItem item in displayToolStripMenuItem.DropDownItems)
            {
                if ((DisplayStyle)item.Tag == style)
                {
                    item.Checked = true;
                }
                else
                {
                    item.Checked = false;
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
            InputHandler.KeyDown(e.KeyCode);

            if (InputHandler.KeyIsDown(Keys.D))
            {
                if (_distLine.Location == new PointF())
                {
                    _distLine.Location = _mouseLocation;
                    _distLine.Location2 = _mouseLocation;
                    _distOver.Location = _mouseLocation.Add(new PointF(30, 5));
                    _distOver.Value = "0.0";
                    _distLine.Show();
                    _distOver.Show();
                }
            }

            switch (e.KeyCode)
            {
                case Keys.ShiftKey:

                    //   MainLoop.WaitForPause();
                    _shiftDown = true;

                    break;

                case Keys.ControlKey:

                    if (!_ctrlDown)
                        MainLoop.WaitForPause();

                    _ctrlDown = true;

                    break;

                case Keys.B:

                    if (_selectedUid != -1)
                    {
                        try
                        {
                            BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].Flag = 2;
                        }
                        catch { }
                    }

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

                case Keys.F9:

                    IO.Serializer.LoadPreviousState();

                    break;
            }
        }

        private void DisplayForm_KeyUp(object sender, KeyEventArgs e)
        {
            InputHandler.KeyUp(e.KeyCode);

            if (!InputHandler.KeyIsDown(Keys.D))
            {
                _distLine.Hide();
                _distOver.Hide();

                _distLine.Location = new PointF();
                _distLine.Location2 = new PointF();
                _distOver.Location = new PointF();
            }

            switch (e.KeyCode)
            {
                case Keys.ShiftKey:

                    _shiftDown = false;
                    // MainLoop.ResumePhysics();

                    break;

                case Keys.ControlKey:

                    _ctrlDown = false;
                    MainLoop.ResumePhysics();

                    break;

                case Keys.P:

                    if (MainLoop.PausePhysics)
                    {
                        MainLoop.ResumePhysics(true);
                    }
                    else
                    {
                        MainLoop.WaitForPause();
                    }

                    break;
            }
        }

        private void RenderBox_MouseDown(object sender, MouseEventArgs e)
        {
            InputHandler.MouseDown(e.Button, e.Location);

            if (e.Button == MouseButtons.Right)
            {
                Cursor.Hide();

                if (!_mouseRightDown)
                {
                    _mouseRightDown = true;

                    MainLoop.WaitForPause();

                    var mUid = MouseOverUID(e.Location);

                    if (mUid != -1)
                    {
                        _mouseId = mUid;
                    }
                    else
                    {
                        _mouseId = BodyManager.Add(ViewportHelpers.ScreenPointToField(e.Location), 1f, ColorHelper.RandomColor());
                    }

                    var bodyPos = ViewportHelpers.FieldPointToScreen(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Position());

                    _flingOver.Location = bodyPos;
                    _flingOver.Location2 = bodyPos;
                    _flingOver.Show();

                    RenderBase.AddOverlay(_flingOver);

                    _orbitOver.Location = bodyPos;
                    _orbitOver.Location2 = bodyPos;
                    _orbitOver.Show();

                    RenderBase.AddOverlay(_orbitOver);
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
            }
        }

        private void RenderBox_MouseUp(object sender, MouseEventArgs e)
        {
            InputHandler.MouseUp(e.Button, e.Location);

            Cursor.Show();

            _bodyMovin = false;

            if (_mouseId != -1)
            {
                _mouseId = -1;
                MainLoop.ResumePhysics();
            }

            if (_ctrlDown && BodyManager.FollowBodyUID != -1)
            {
                BodyManager.FollowSelected = true;
            }

            if (_mouseRightDown)
            {
                _mouseRightDown = false;

                _flingOver.Hide();
                _orbitOver.Hide();
            }

            _flingStartPos = new PointF();
            _flingPrevScreenPos = new Point();
        }

        private void RenderBox_MouseMove(object sender, MouseEventArgs e)
        {
            InputHandler.MouseMove(e.Location);

            _mouseLocation = e.Location;

            if (e.Button == MouseButtons.Left)
            {
                if (_selectedUid != -1 && _shiftDown)
                {
                    _bodyMovin = true;
                }

                if (_bodyMovin)
                {
                    BodyManager.Move(BodyManager.UIDToIndex(_selectedUid), ViewportHelpers.ScreenPointToField(e.Location));
                }
                else
                {
                    var moveDiff = e.Location.Subtract(_mouseMoveDownLoc);
                    ViewportOffsets.ViewportOffset = ViewportOffsets.ViewportOffset.Add(ViewportHelpers.FieldPointToScreenNoOffset(moveDiff));
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
                    _flingOver.Location2 = _flingOver.Location.Subtract(deflection);

                    // Flip and shorten the vector and apply it to the body speed.
                    BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].VeloX = -deflection.X / 3f;
                    BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].VeloY = -deflection.Y / 3f;

                    // Calculate the true screen position from the body location.
                    var clientPosition = ViewportHelpers.FieldPointToScreen(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Position());
                    var screenPosition = RenderBox.PointToScreen(clientPosition.ToPoint());

                    // Lock the cursor in place above the body.
                    Cursor.Position = screenPosition;
                    _flingPrevScreenPos = screenPosition;

                    // Calculate the new orbital path.
                    var orbitPath = BodyManager.CalcPathCircle(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)]);

                    // Update the orbit overlay.
                    _orbitOver.Location = new PointF(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].PosX, BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].PosY);
                    _orbitOver.OrbitPath = orbitPath;
                    _orbitOver.Show();
                    RenderBase.AddOverlay(_orbitOver);
                }

            }

            if (InputHandler.KeyIsDown(Keys.D))
            {
                _distLine.Location2 = _mouseLocation;
                _distOver.Location = _mouseLocation.Add(new PointF(30, 5));

                var loc1 = ViewportHelpers.ScreenPointToField(_distLine.Location);
                var loc2 = ViewportHelpers.ScreenPointToField(_distLine.Location2);

                _distOver.Value = loc1.DistanceSqrt(loc2).ToString();
            }
        }

        private void RenderBox_MouseWheel(object sender, MouseEventArgs e)
        {
            InputHandler.MouseWheel(e.Delta);

            var scaleChange = 0.05f * ViewportOffsets.CurrentScale;
            float newScale = ViewportOffsets.CurrentScale;

            if (e.Delta > 0)
            {
                newScale += scaleChange;

                if (!InputHandler.KeysDown && !InputHandler.MouseIsDown)
                    ViewportOffsets.Zoom(newScale, e.Location);

                if (_mouseRightDown && _mouseId != -1)
                {
                    BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size += 1.0f;
                    BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Mass = BodyManager.CalcMass(BodyManager.Bodies[BodyManager.UIDToIndex(_mouseId)].Size);
                }
            }
            else
            {
                newScale -= scaleChange;

                if (!InputHandler.KeysDown && !InputHandler.MouseIsDown)
                    ViewportOffsets.Zoom(newScale, e.Location);

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
            ViewportOffsets.ScreenCenter = new PointF(this.RenderBox.Width * 0.5f, this.RenderBox.Height * 0.5f);
        }

        private void AddBodiesButton_Click(object sender, EventArgs e)
        {
            if (_addFormInstance == null || _addFormInstance.IsDisposed)
            {
                _addFormInstance = new AddBodiesForm();
            }

            _addFormInstance.WindowState = FormWindowState.Normal;
            _addFormInstance.Activate();
            _addFormInstance.Show();
        }

        private void TrailsCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            RenderBase.Trails = TrailsCheckBox.Checked;
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
            _selectedUid = -1;
            _mouseId = -1;
            _bodyMovin = false;
            BodyManager.ClearBodies();
            MainLoop.ResumePhysics();
        }

        private void normalToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SetDisplayStyle(DisplayStyle.Normal);
        }

        private void pressuresToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SetDisplayStyle(DisplayStyle.Pressures);
        }

        private void highContrastToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            SetDisplayStyle(DisplayStyle.HighContrast);
        }

        private void speedsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SetDisplayStyle(DisplayStyle.Speeds);
        }

        private void indexToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SetDisplayStyle(DisplayStyle.Index);
        }

        private void forcesToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SetDisplayStyle(DisplayStyle.Forces);
        }

        private void UpdateButton_Click(object sender, EventArgs e)
        {
            if (_selectedUid != -1)
            {
                BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].VeloX = Convert.ToSingle(VeloXTextBox.Text.Trim());
                BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].VeloY = Convert.ToSingle(VeloYTextBox.Text.Trim());
                BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].Size = Convert.ToSingle(RadiusTextBox.Text.Trim());
                BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].Mass = Convert.ToSingle(MassTextBox.Text.Trim());
                BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].Flag = Convert.ToInt32(FlagsTextBox.Text.Trim());
            }
        }

        private void PressureScaleUpDown_ValueChanged(object sender, EventArgs e)
        {
            RenderBase.StyleScaleMax = (float)PressureScaleUpDown.Value;
        }

        private void CenterOnMassButton_Click(object sender, EventArgs e)
        {
            var cm = BodyManager.CenterOfMass().Multi(-1.0f);

            ViewportOffsets.ViewportOffset = cm;
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
            MainLoop.End();
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

        private void showMeshToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            RenderBase.ShowMesh = showMeshToolStripMenuItem.Checked;
        }

        private void allForceVectorsToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            RenderBase.ShowAllForce = allForceVectorsToolStripMenuItem.Checked;
        }

        private void sortZOrderToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            RenderBase.SortZOrder = sortZOrderToolStripMenuItem.Checked;
        }

        private void fastPrimitivesToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            RenderBase.FastPrimitives = fastPrimitivesToolStripMenuItem.Checked;
        }

        private void rewindBufferToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            MainLoop.RewindBuffer = rewindBufferToolStripMenuItem.Checked;
        }

        private void collisionsToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            MainLoop.Collisions = collisionsToolStripMenuItem.Checked;
        }
    }
}