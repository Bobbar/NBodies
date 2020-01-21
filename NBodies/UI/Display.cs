﻿using NBodies.Extensions;
using NBodies.Physics;
using NBodies.Rendering;
using NBodies.UI.KeyActions;
using NBodies.Helpers;
using System;
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

        private Camera _camera;
        private bool _firstMove = true;
        private Vector2 _lastPos;
        private int _vertexBufferObject;
        private int _colorBufferObject;
        private int _cubeVertBufferObject;
        private int _cubeVertArrayObject;

        private int _vertexArrayObject;
        private Shader _shader;
        private Vertex[] _verts = new Vertex[0];


        private readonly float[] _cubeVerts =
       {
             // Position
            -1.0f, -1.0f, -1.0f, // Front face
             1.0f, -1.0f, -1.0f,
             1.0f,  1.0f, -1.0f,
             1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
            -1.0f, -1.0f, -1.0f,

            -1.0f, -1.0f,  1.0f, // Back face
             1.0f, -1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,

            -1.0f,  1.0f,  1.0f, // Left face
            -1.0f,  1.0f, -1.0f,
            -1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,

             1.0f,  1.0f,  1.0f, // Right face
             1.0f,  1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,

            -1.0f, -1.0f, -1.0f, // Bottom face
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f,  1.0f,
             1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f, -1.0f,

            -1.0f,  1.0f, -1.0f, // Top face
             1.0f,  1.0f, -1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f, -1.0f
        };


        //private readonly float[] _cubeVerts =
        //{
        //     // Position
        //    -0.5f, -0.5f, -0.5f, // Front face
        //     0.5f, -0.5f, -0.5f,
        //     0.5f,  0.5f, -0.5f,
        //     0.5f,  0.5f, -0.5f,
        //    -0.5f,  0.5f, -0.5f,
        //    -0.5f, -0.5f, -0.5f,

        //    -0.5f, -0.5f,  0.5f, // Back face
        //     0.5f, -0.5f,  0.5f,
        //     0.5f,  0.5f,  0.5f,
        //     0.5f,  0.5f,  0.5f,
        //    -0.5f,  0.5f,  0.5f,
        //    -0.5f, -0.5f,  0.5f,

        //    -0.5f,  0.5f,  0.5f, // Left face
        //    -0.5f,  0.5f, -0.5f,
        //    -0.5f, -0.5f, -0.5f,
        //    -0.5f, -0.5f, -0.5f,
        //    -0.5f, -0.5f,  0.5f,
        //    -0.5f,  0.5f,  0.5f,

        //     0.5f,  0.5f,  0.5f, // Right face
        //     0.5f,  0.5f, -0.5f,
        //     0.5f, -0.5f, -0.5f,
        //     0.5f, -0.5f, -0.5f,
        //     0.5f, -0.5f,  0.5f,
        //     0.5f,  0.5f,  0.5f,

        //    -0.5f, -0.5f, -0.5f, // Bottom face
        //     0.5f, -0.5f, -0.5f,
        //     0.5f, -0.5f,  0.5f,
        //     0.5f, -0.5f,  0.5f,
        //    -0.5f, -0.5f,  0.5f,
        //    -0.5f, -0.5f, -0.5f,

        //    -0.5f,  0.5f, -0.5f, // Top face
        //     0.5f,  0.5f, -0.5f,
        //     0.5f,  0.5f,  0.5f,
        //     0.5f,  0.5f,  0.5f,
        //    -0.5f,  0.5f,  0.5f,
        //    -0.5f,  0.5f, -0.5f
        //};


        public DisplayForm()
        {
            InitializeComponent();

            _UIUpdateTimer.Interval = 100;//250;
            _UIUpdateTimer.Tick += _UIUpdateTimer_Tick;
            //_UIUpdateTimer.Start();

            RenderBox.MouseWheel += RenderBox_MouseWheel;

            TimeStepUpDown.Value = (decimal)MainLoop.TimeStep;
            StyleScaleUpDown.Value = (decimal)RenderBase.StyleScaleMax;
            AlphaUpDown.Value = RenderBase.BodyAlpha;

            RenderBox.DoubleBuffered(true);
        }

        private void DisplayForm_Load(object sender, EventArgs e)
        {
            ViewportOffsets.ScreenCenter = new PointF(this.RenderBox.Width / 2f, this.RenderBox.Height / 2f);
            ViewportOffsets.ScaleOffset = ViewportHelpers.FieldPointToScreenNoOffset(ViewportOffsets.ScreenCenter);
            MainLoop.MaxThreadsPerBlock = Program.ThreadsPerBlockArgument;

            using (var selectDevice = new ChooseDeviceForm())
            {
                var result = selectDevice.ShowDialog();

                if (result == DialogResult.OK)
                {
                    var device = selectDevice.SelectedDevice;
                    var threads = selectDevice.MaxThreadsPerBlock;

                    PhysicsProvider.InitPhysics(device, threads);

                }
                else
                {
                    Application.Exit();
                }
            }

            _camera = new Camera(Vector3.UnitZ, glControl.ClientSize.Width / (float)glControl.ClientSize.Height);

            glControl.MouseDown += GlControl_MouseDown;
            glControl.MouseUp += GlControl_MouseUp; ;
            glControl.MouseWheel += GlControl_MouseWheel;
            glControl.MouseMove += GlControl_MouseMove;
            glControl.Paint += GlControl_Paint;
            glControl.KeyDown += GlControl_KeyDown;
            glControl.KeyUp += GlControl_KeyUp;
            glControl.Resize += GlControl_Resize;
            glControl.MakeCurrent();
            MainLoop.GLRenderer = glControl;


            GL.PointSize(5.0f);

            //GL.ClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            GL.ClearColor(Color.Black);

            GL.Enable(EnableCap.DepthTest);

            _shader = new Shader(Environment.CurrentDirectory + $@"/Rendering/Shaders/shader.vert", Environment.CurrentDirectory + $@"/Rendering/Shaders/shader.frag");
            _shader.BindFragDataLocation(0, "FragColor");
            _shader.BindAttribLocation(0, "aPosition");
            _shader.BindAttribLocation(1, "aObjColor");
            _shader.Use();

            _vertexBufferObject = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vertexBufferObject);
            GL.BufferData(BufferTarget.ArrayBuffer, _verts.Length * sizeof(float), _verts, BufferUsageHint.StaticDraw);

            _vertexArrayObject = GL.GenVertexArray();
            GL.BindVertexArray(_vertexArrayObject);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vertexBufferObject);

            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 28, 0);
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(1, 4, VertexAttribPointerType.Float, false, 28, 12);
            GL.EnableVertexAttribArray(1);
            GL.BindVertexArray(0);



            _cubeVertBufferObject = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, _cubeVertBufferObject);
            GL.BufferData(BufferTarget.ArrayBuffer, _cubeVerts.Length * sizeof(float), _cubeVerts, BufferUsageHint.StaticDraw);

            _cubeVertArrayObject = GL.GenVertexArray();
            GL.BindVertexArray(_cubeVertArrayObject);
            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), 0);
            GL.EnableVertexAttribArray(0);
            //GL.VertexAttribPointer(1, 4, VertexAttribPointerType.Float, false, 28, 12);
            //GL.EnableVertexAttribArray(1);
            GL.BindVertexArray(0);


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

            //    NBodies.IO.Serializer.LoadPreviousState();

            //    MainLoop.StartLoop();

            _UIUpdateTimer.Start();
        }

        private void GlControl_MouseUp(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            InputHandler.MouseUp(e.Button, e.Location);
        }

        private void GlControl_Resize(object sender, EventArgs e)
        {
            _camera.AspectRatio = glControl.ClientSize.Width / (float)glControl.ClientSize.Height;
            GL.Viewport(glControl.ClientSize);
        }

        private void GlControl_KeyUp(object sender, KeyEventArgs e)
        {
            InputHandler.KeyUp(e.KeyCode);

        }

        private void GlControl_KeyDown(object sender, KeyEventArgs e)
        {
            InputHandler.KeyDown(e.KeyCode);

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

        private void GlControl_Paint(object sender, PaintEventArgs e)
        {
            const float cameraSpeed = 200f;//1.5f;
            const float time = 0.016f;

            if (InputHandler.KeyIsDown(Keys.W))
                _camera.Position += _camera.Front * cameraSpeed * time; // Forward 
            if (InputHandler.KeyIsDown(Keys.S))
                _camera.Position -= _camera.Front * cameraSpeed * time; // Backwards
            if (InputHandler.KeyIsDown(Keys.A))
                _camera.Position -= _camera.Right * cameraSpeed * time; // Left
            if (InputHandler.KeyIsDown(Keys.D))
                _camera.Position += _camera.Right * cameraSpeed * time; // Right
            if (InputHandler.KeyIsDown(Keys.Space))
                _camera.Position += _camera.Up * cameraSpeed * time; // Up 
            if (InputHandler.KeyIsDown(Keys.ShiftKey))
                _camera.Position -= _camera.Up * cameraSpeed * time; // Down

            //   Console.WriteLine($@"{_camera.Position.ToString()}");

            // Render Bodies
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);


            var bodies = BodyManager.Bodies;

            if (bodies.Length == 0)
                return;

            //_verts = new Vertex[bodies.Length];

            //for (int i = 0; i < bodies.Length; i++)
            //{
            //    var b = bodies[i];
            //    var bColor = Color.FromArgb(bodies[i].Color);

            //    _verts[i].Position = new Vector3(b.PosX, b.PosY, b.PosZ);
            //    //_verts[i].Color = new Color4(bColor.R, bColor.G, bColor.B, 200);
            //    _verts[i].Color = RenderBase.GetVariableColor(Color.Blue, Color.Red, Color.Yellow, bodies.Length, i, true);

            //}



            //GL.BindBuffer(BufferTarget.ArrayBuffer, _vertexBufferObject);
            //GL.BufferData(BufferTarget.ArrayBuffer, _verts.Length * 28, _verts, BufferUsageHint.StaticDraw);
            //GL.BindVertexArray(_vertexArrayObject);

            //_shader.Use();
            //var model = Matrix4.Identity;// * Matrix4.CreateRotationX((float)MathHelper.DegreesToRadians(_time));
            //_shader.SetMatrix4("model", model);
            //_shader.SetMatrix4("view", _camera.GetViewMatrix());
            //_shader.SetMatrix4("projection", _camera.GetProjectionMatrix());
            //_shader.SetInt("isMesh", 0);


            //GL.DrawArrays(PrimitiveType.Points, 0, _verts.Length);

            //  Draw Body cubes.
            GL.BindVertexArray(0);
            GL.BindVertexArray(_cubeVertArrayObject);

            //GL.Enable(EnableCap.LineSmooth); // This is Optional 
            //                                 //      GL.Enable(EnableCap.Normalize);  
            //GL.Enable(EnableCap.RescaleNormal);


            //GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
            //GL.Enable(EnableCap.PolygonOffsetFill);

            //GL.Enable(EnableCap.Blend);
            //GL.DepthMask(false);

            

            for (int i = 0; i < bodies.Length; i++)
            {
                var body = bodies[i];

                _shader.Use();
                var meshModel = Matrix4.Identity;
                meshModel *= Matrix4.CreateScale(body.Size / 2);
                meshModel *= Matrix4.CreateTranslation(body.PosX, body.PosY, body.PosZ);

                _shader.SetMatrix4("model", meshModel);
                _shader.SetMatrix4("view", _camera.GetViewMatrix());
                _shader.SetMatrix4("projection", _camera.GetProjectionMatrix());
                _shader.SetInt("isMesh", 1);
                var bColor = Color.FromArgb(body.Color);
                var styleColor = RenderBase.GetVariableColor(Color.Blue, Color.Red, Color.Yellow, RenderBase.StyleScaleMax, body.Pressure, true);
                //_shader.SetVector3("meshColor", new Vector3(bColor.R / 255f, bColor.G / 255f, bColor.B / 255f));
                _shader.SetVector3("meshColor", new Vector3(styleColor.R / 255f, styleColor.G / 255f, styleColor.B / 255f));

                GL.DrawArrays(PrimitiveType.Triangles, 0, _cubeVerts.Length);
            }
            GL.BindVertexArray(0);

            //GL.Disable(EnableCap.Blend);
            //GL.DepthMask(true);

            if (RenderBase.ShowMesh)
            {
                //  Draw mesh
                if (BodyManager.Mesh.Length > 1)
                {
                    GL.BindVertexArray(0);
                    GL.BindVertexArray(_cubeVertArrayObject);

                    //   GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);

                    for (int m = 0; m < BodyManager.Mesh.Length; m++)
                    {
                        var mesh = BodyManager.Mesh[m];

                        _shader.Use();
                        var meshModel = Matrix4.Identity;
                        meshModel *= Matrix4.CreateScale(mesh.Size / 2);
                        meshModel *= Matrix4.CreateTranslation(mesh.LocX, mesh.LocY, mesh.LocZ);

                        _shader.SetMatrix4("model", meshModel);
                        _shader.SetMatrix4("view", _camera.GetViewMatrix());
                        _shader.SetMatrix4("projection", _camera.GetProjectionMatrix());
                        _shader.SetInt("isMesh", 1);
                        GL.DrawArrays(PrimitiveType.LineLoop, 0, _cubeVerts.Length);
                    }
                    GL.BindVertexArray(0);
                }
            }






            glControl.SwapBuffers();



            // GL.BindVertexArray(0);
        }



        private void GlControl_MouseDown(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            InputHandler.MouseDown(e.Button, e.Location);

            if (e.Button == MouseButtons.Right)
                _lastPos = new Vector2(e.X, e.Y);
        }


        private void GlControl_MouseMove(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            InputHandler.MouseMove(e.Location);

            const float sensitivity = 0.2f;

            // var mouse = Mouse.GetState();

            //    Console.WriteLine($@"M: {e.Location.ToString()}   {mouse.X},{mouse.Y}   Conn:{mouse.IsConnected}");

            if (e.Button == MouseButtons.Right)
            {
                if (_firstMove) // this bool variable is initially set to true
                {
                    _lastPos = new Vector2(e.X, e.Y);
                    _firstMove = false;
                }
                else
                {
                    // Calculate the offset of the mouse position
                    var deltaX = e.X - _lastPos.X;
                    var deltaY = e.Y - _lastPos.Y;
                    _lastPos = new Vector2(e.X, e.Y);

                    // Apply the camera pitch and yaw (we clamp the pitch in the camera class)
                    _camera.Yaw += deltaX * sensitivity;
                    _camera.Pitch -= deltaY * sensitivity; // reversed since y-coordinates range from bottom to top
                }

                //Cursor.Position = e.Location;
                //    Console.WriteLine($@"{_camera.Yaw}  {_camera.Pitch}");
            }



        }

        private void GlControl_MouseWheel(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            InputHandler.MouseWheel(e.Delta);

            if (e.Delta > 0)
                _camera.Fov -= 1;
            else
                _camera.Fov -= -1;
            //  Console.WriteLine(_camera.Fov);
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
                //  MainLoop.Renderer = new D2DRenderer(RenderBox);
                MainLoop.Renderer = new OpenTKRenderer(RenderBox);
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

            //  MainLoop.Renderer.DrawBodiesAsync(BodyManager.Bodies, true, null);



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

            if (_selectedUid != -1 && !MainLoop.PausePhysics)
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
            if (_selectedUid != -1)
            {
                var selectBody = BodyManager.BodyFromUID(_selectedUid);

                VeloXTextBox.Text = selectBody.VeloX.ToString();
                VeloYTextBox.Text = selectBody.VeloY.ToString();
                RadiusTextBox.Text = selectBody.Size.ToString();
                MassTextBox.Text = selectBody.Mass.ToString();
                FlagsTextBox.Text = selectBody.Flag.ToString();

                selectBody.PrintInfo();
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
                            BodyManager.Bodies[BodyManager.UIDToIndex(_selectedUid)].InRoche = 1;
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

        //private void DisplayForm_KeyUp(object sender, KeyEventArgs e)
        //{
        //    InputHandler.KeyUp(e.KeyCode);

        //    if (!InputHandler.KeyIsDown(Keys.D))
        //    {
        //        _distLine.Hide();
        //        _distOver.Hide();

        //        _distLine.Location = new PointF();
        //        _distLine.Location2 = new PointF();
        //        _distOver.Location = new PointF();
        //    }

        //    switch (e.KeyCode)
        //    {
        //        case Keys.ShiftKey:
        //            _shiftDown = false;

        //            break;

        //        case Keys.ControlKey:
        //            _ctrlDown = false;
        //            MainLoop.ResumePhysics();

        //            break;

        //        case Keys.P:

        //            if (MainLoop.PausePhysics)
        //            {
        //                MainLoop.ResumePhysics(true);
        //            }
        //            else
        //            {
        //                MainLoop.WaitForPause();
        //            }

        //            break;
        //    }
        //}

        private void RenderBox_MouseDown(object sender, System.Windows.Forms.MouseEventArgs e)
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

        private void RenderBox_MouseUp(object sender, System.Windows.Forms.MouseEventArgs e)
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

        private void RenderBox_MouseMove(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            InputHandler.MouseMove(e.Location);
            _mouseLocation = e.Location;

            if (InputHandler.KeyIsDown(Keys.D))
            {
                _distLine.Location2 = _mouseLocation;
                _distOver.Location = _mouseLocation.Add(new PointF(30, 5));

                var loc1 = ViewportHelpers.ScreenPointToField(_distLine.Location);
                var loc2 = ViewportHelpers.ScreenPointToField(_distLine.Location2);

                _distOver.Value = loc1.DistanceSqrt(loc2).ToString();
            }


            if (!InputHandler.MouseIsDown)
                return;

            if (e.Button == MouseButtons.Left)
            {
                if (_selectedUid != -1 && _shiftDown)
                {
                    _bodyMovin = true;
                }

                if (_bodyMovin)
                {
                    var loc = ViewportHelpers.ScreenPointToField(e.Location);
                    if (snapToGridToolStripMenuItem.Checked)
                        loc = loc.SnapToGrid(2);
                    BodyManager.Move(BodyManager.UIDToIndex(_selectedUid), loc);
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


        }

        private void RenderBox_MouseWheel(object sender, System.Windows.Forms.MouseEventArgs e)
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
            _bodyMovin = false;
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
            var cm = BodyManager.CenterOfMass3D();
            _camera.Position = cm;


            // ViewportHelpers.CenterCurrentField();
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

        private void syncRendererToolStripMenuItem_CheckedChanged(object sender, EventArgs e)
        {
            MainLoop.SyncRenderer = syncRendererToolStripMenuItem.Checked;
        }
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct Vertex
    {
        public Vector3 Position;
        public Color4 Color;
    }
}