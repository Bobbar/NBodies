using NBodies.Extensions;
using NBodies.Physics;
using NBodies.Rendering;
using NBodies.UI.KeyActions;
using NBodies.Helpers;
using System;
using System.Diagnostics;
using System.Drawing;
using System.Windows.Forms;
using OpenTK;
using OpenTK.Input;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL4;
using System.Runtime.InteropServices;

namespace NBodies.UI
{
    public partial class DisplayForm : Form
    {
        private AddBodiesForm _addFormInstance = new AddBodiesForm();
        private PlaybackControlForm _playbackControl;
        private Stopwatch _timer = new Stopwatch();
        private bool _ctrlDown = false;
        private bool _hideToolbar = false;
        private float _ogToolbarHeight;

        private int _selectedUid = -1;
        private int _mouseId = -1;
        private PointF _mouseLocation = new PointF();

        private Timer _UIUpdateTimer = new Timer();

        private OverlayGraphic _flingOver = new OverlayGraphic(OverlayGraphicType.Line, new PointF(), "");
        private OverlayGraphic _orbitOver = new OverlayGraphic(OverlayGraphicType.Orbit, new PointF(), "");
        private OverlayGraphic _distLine = new OverlayGraphic(OverlayGraphicType.Line, new PointF(), "");
        private OverlayGraphic _distOver = new OverlayGraphic(OverlayGraphicType.Text, new PointF(), "");

        public DisplayForm()
        {
            InitializeComponent();

            _UIUpdateTimer.Interval = 100;
            _UIUpdateTimer.Tick += _UIUpdateTimer_Tick;

            TimeStepUpDown.Value = (decimal)MainLoop.TimeStep;
            StyleScaleUpDown.Value = (decimal)RenderBase.StyleScaleMax;
            AlphaUpDown.Value = RenderBase.BodyAlpha;
        }

        private void DisplayForm_Load(object sender, EventArgs e)
        {
            RenderBase.SetStyleScales();
            ViewportOffsets.ScreenCenter = new PointF(this.glControl.Width / 2f, this.glControl.Height / 2f);
            ViewportOffsets.ScaleOffset = ViewportHelpers.FieldPointToScreenNoOffset(ViewportOffsets.ScreenCenter);
            MainLoop.MaxThreadsPerBlock = Program.ThreadsPerBlockArgument;

            using (var selectDevice = new ChooseDeviceForm())
            {
                var result = selectDevice.ShowDialog();

                if (result == DialogResult.OK)
                {
                    var device = selectDevice.SelectedDevice;
                    var threads = selectDevice.MaxThreadsPerBlock;
                    var fastMath = selectDevice.FastMath;

                    PhysicsProvider.InitPhysics(device, threads, fastMath);

                }
                else
                {
                    Application.Exit();
                }
            }

         
            glControl.KeyDown += GlControl_KeyDown;
            MainLoop.GLRenderer = glControl;

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
            InputHandler.AddKeyAction(new KernelSizeKey());
            InputHandler.AddKeyAction(new ZeroVeloKey());
            InputHandler.AddKeyAction(new GasKKey());

            PopulateDisplayStyleMenu();

            MainLoop.StartLoop();

            _UIUpdateTimer.Start();
        }

        private void GlControl_KeyDown(object sender, KeyEventArgs e)
        {
            if (InputHandler.KeyIsDown(Keys.F11))
            {
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
            }

            if (InputHandler.KeyIsDown(Keys.P))
            {
                if (MainLoop.PausePhysics)
                {
                    MainLoop.ResumePhysics(true);
                }
                else
                {
                    MainLoop.WaitForPause();
                }
            }

            if (InputHandler.KeyIsDown(Keys.F9))
                IO.Serializer.LoadPreviousState();
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

            AlphaUpDown.Value = RenderBase.BodyAlpha;
            TimeStepUpDown.Value = (decimal)MainLoop.TimeStep;
            StyleScaleUpDown.Value = (decimal)RenderBase.StyleScaleMax;
            SetDisplayStyle(RenderBase.DisplayStyle);

            if (BodyManager.FollowBodyUID != -1 && !MainLoop.PausePhysics)
            {
                SetSelectedInfo();
            }

            if (MainLoop.Recorder.RecordingActive)
            {
                RecordButton.BackColor = Color.DarkGreen;
            }
            else
            {
                RecordButton.BackColor = DefaultBackColor;
            }
        }

        private void SetSelectedInfo()
        {
            if (BodyManager.FollowBodyUID != -1)
            {
                var selectBody = BodyManager.FollowBody();

                VeloXTextBox.Text = selectBody.VeloX.ToString();
                VeloYTextBox.Text = selectBody.VeloY.ToString();
                RadiusTextBox.Text = selectBody.Size.ToString();
                MassTextBox.Text = selectBody.Mass.ToString();
                FlagsTextBox.Text = selectBody.Flag.ToString();
#if DEBUG
               // selectBody.PrintInfo();
#endif
            }
        }

        private void StartRecording()
        {
            if (MainLoop.Recording)
            {
                MainLoop.StopRecording();
            }


            using (var settingsForm = new RecordSettings())
            using (var saveDialog = new SaveFileDialog())
            {
                float timeStep;
                double maxSize;

                settingsForm.ShowDialog();

                timeStep = settingsForm.TimeStep;
                maxSize = settingsForm.MaxRecordSize;

                saveDialog.Filter = "NBody Recording|*.rec";
                saveDialog.Title = "Save Recording";
                saveDialog.ShowDialog();

                if (!string.IsNullOrEmpty(saveDialog.FileName))
                {
                    MainLoop.StartRecording(saveDialog.FileName, timeStep, maxSize);
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

        private void PopulateDisplayStyleMenu()
        {
            var styles = Enum.GetValues(typeof(DisplayStyle));

            foreach (DisplayStyle s in styles)
            {
                string name = Enum.GetName(typeof(DisplayStyle), s);
                var styleTool = new ToolStripMenuItem(name);
                styleTool.Tag = s;
                styleTool.CheckOnClick = true;
                styleTool.Click += StyleTool_Click;
                displayToolStripMenuItem.DropDownItems.Add(styleTool);
            }
        }

        private void StyleTool_Click(object sender, EventArgs e)
        {
            var styleTool = sender as ToolStripMenuItem;
            DisplayStyle style = (DisplayStyle)styleTool.Tag;
            SetDisplayStyle(style);
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

        private void PauseButton_Click(object sender, EventArgs e)
        {
            if (!MainLoop.PausePhysics)
                MainLoop.WaitForPause();
            else
                MainLoop.ResumePhysics(true);
        }

        private void saveStateToolStripMenuItem_Click(object sender, EventArgs e)
        {
            NBodies.IO.Serializer.SaveState();
        }

        private void loadStateToolStripMenuItem_Click(object sender, EventArgs e)
        {
            NBodies.IO.Serializer.LoadState();
            ViewportHelpers.CenterCurrentField();
        }

        private void reloadPreviousToolStripMenuItem_Click(object sender, EventArgs e)
        {
            NBodies.IO.Serializer.LoadPreviousState();
            ViewportHelpers.CenterCurrentField();
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
            BodyManager.ClearBodies();
            PhysicsProvider.PhysicsCalc.Flush();
            MainLoop.ResumePhysics();
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

        private void StyleScaleUpDown_ValueChanged(object sender, EventArgs e)
        {
            RenderBase.StyleScaleMax = (float)StyleScaleUpDown.Value;
        }

        private void CenterOnMassButton_Click(object sender, EventArgs e)
        {
            glControl.MoveCameraToCenterMass();
        }

        private void ToggleRendererButton_Click(object sender, EventArgs e)
        {
          //  SwitchRenderer();
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

        private void usePointsToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            glControl.UsePoints = usePointsToolStripMenuItem.Checked;
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

        private void syncRendererToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            MainLoop.SyncRenderer = syncRendererToolStripMenuItem.Checked;
        }
    }
}