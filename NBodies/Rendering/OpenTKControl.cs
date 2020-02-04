using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL4;
using NBodies;
using NBodies.Extensions;
using NBodies.Physics;
using NBodies.Rendering;
using NBodies.UI;
using NBodies.UI.KeyActions;
using NBodies.Helpers;
using System.Drawing;
using System.Diagnostics;

namespace NBodies.Rendering
{
    public class OpenTKControl : GLControl
    {
        // Quad based cube.
        private readonly float[] _cubeVerts =
        {
            // front
            -1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f, 1.0f, -1.0f,
             -1.0f, 1.0f, -1.0f,
            // back
             -1.0f, -1.0f, 1.0f,
             1.0f, -1.0f, 1.0f,
             1.0f, 1.0f, 1.0f,
             -1.0f, 1.0f, 1.0f,
            // right
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, 1.0f,
             1.0f, 1.0f, 1.0f,
             1.0f, 1.0f, -1.0f,
            // left
             -1.0f, -1.0f, -1.0f,
             -1.0f, -1.0f, 1.0f,
             -1.0f, 1.0f, 1.0f,
             -1.0f, 1.0f, -1.0f,
            // top
             -1.0f, 1.0f, -1.0f,
             1.0f, 1.0f, -1.0f,
             1.0f, 1.0f, 1.0f,
             -1.0f, 1.0f, 1.0f,
            // bottom
             -1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, 1.0f,
             -1.0f, -1.0f, 1.0f
        };

        // Quad based cube normals.
        private readonly float[] _normalVerts =
        {
            // front
            0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, -1.0f, 
            // back
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 
            // right
            1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 
            // left
            -1.0f, 0.0f, 0.0f,
            -1.0f, 0.0f, 0.0f,
            -1.0f, 0.0f, 0.0f,
            -1.0f, 0.0f, 0.0f, 
            // top
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            // bottom
            0.0f, -1.0f, 0.0f,
            0.0f, -1.0f, 0.0f,
            0.0f, -1.0f, 0.0f,
            0.0f, -1.0f, 0.0f,
        };

        #region Triangle based cube verts
        // Triangle based cube.
        // private readonly float[] _cubeVerts =
        //{
        //          // Position
        //         -1.0f, -1.0f, -1.0f, // Front face
        //          1.0f, -1.0f, -1.0f,
        //          1.0f,  1.0f, -1.0f,
        //          1.0f,  1.0f, -1.0f,
        //         -1.0f,  1.0f, -1.0f,
        //         -1.0f, -1.0f, -1.0f,

        //         -1.0f, -1.0f,  1.0f, // Back face
        //          1.0f, -1.0f,  1.0f,
        //          1.0f,  1.0f,  1.0f,
        //          1.0f,  1.0f,  1.0f,
        //         -1.0f,  1.0f,  1.0f,
        //         -1.0f, -1.0f,  1.0f,

        //         -1.0f,  1.0f,  1.0f, // Left face
        //         -1.0f,  1.0f, -1.0f,
        //         -1.0f, -1.0f, -1.0f,
        //         -1.0f, -1.0f, -1.0f,
        //         -1.0f, -1.0f,  1.0f,
        //         -1.0f,  1.0f,  1.0f,

        //          1.0f,  1.0f,  1.0f, // Right face
        //          1.0f,  1.0f, -1.0f,
        //          1.0f, -1.0f, -1.0f,
        //          1.0f, -1.0f, -1.0f,
        //          1.0f, -1.0f,  1.0f,
        //          1.0f,  1.0f,  1.0f,

        //         -1.0f, -1.0f, -1.0f, // Bottom face
        //          1.0f, -1.0f, -1.0f,
        //          1.0f, -1.0f,  1.0f,
        //          1.0f, -1.0f,  1.0f,
        //         -1.0f, -1.0f,  1.0f,
        //         -1.0f, -1.0f, -1.0f,

        //         -1.0f,  1.0f, -1.0f, // Top face
        //          1.0f,  1.0f, -1.0f,
        //          1.0f,  1.0f,  1.0f,
        //          1.0f,  1.0f,  1.0f,
        //         -1.0f,  1.0f,  1.0f,
        //         -1.0f,  1.0f, -1.0f
        //     };

        // Triangle based cube normals.
        //      private readonly float[] _normalVerts =
        //{
        //          // front
        //          0.0f, 0.0f, -1.0f,
        //          0.0f, 0.0f, -1.0f,
        //          0.0f, 0.0f, -1.0f,
        //          0.0f, 0.0f, -1.0f, 
        //          // back
        //          0.0f, 0.0f, 1.0f,
        //          0.0f, 0.0f, 1.0f,
        //          0.0f, 0.0f, 1.0f,
        //          0.0f, 0.0f, 1.0f, 
        //            // left
        //          -1.0f, 0.0f, 0.0f,
        //          -1.0f, 0.0f, 0.0f,
        //          -1.0f, 0.0f, 0.0f,
        //          -1.0f, 0.0f, 0.0f, 
        //          // right
        //          1.0f, 0.0f, 0.0f,
        //          1.0f, 0.0f, 0.0f,
        //          1.0f, 0.0f, 0.0f,
        //          1.0f, 0.0f, 0.0f, 
        //        // bottom
        //          0.0f, -1.0f, 0.0f,
        //          0.0f, -1.0f, 0.0f,
        //          0.0f, -1.0f, 0.0f,
        //          0.0f, -1.0f, 0.0f,
        //          // top
        //          0.0f, 1.0f, 0.0f,
        //          0.0f, 1.0f, 0.0f,
        //          0.0f, 1.0f, 0.0f,
        //          0.0f, 1.0f, 0.0f,

        //      };
        #endregion Triangle based cube verts

        private int _cubeVertBufferObject;
        private int _cubeVertArrayObject;

        private int _normVertBufferObject;
        private int _normVertArrayObject;

        private int _offsetBufferObject;
        private int _offsetArrayObject;

        private int _colorBufferObject;
        private int _colorArrayObject;

        private Vector4[] _offsets = new Vector4[0];
        private Vector3[] _colors = new Vector3[0];

        private int _normAttrib;
        private int _posAttrib;
        private int _offsetAttrib;
        private int _colorAttrib;

        private Camera _camera;
        const float cameraSpeedFast = 200f;
        const float cameraSpeedSlow = 10f;
        private float cameraSpeed = cameraSpeedFast;

        private bool _firstMove = true;
        private Vector2 _lastPos;

        private Shader _shader;

        private float[] _orderDist = new float[0];
        private int[] _orderIdx = new int[0];

        public OpenTKControl(GraphicsMode mode) : base(mode)
        {
            _camera = new Camera(Vector3.UnitZ, this.ClientSize.Width / (float)this.ClientSize.Height);

            this.MakeCurrent();

            GL.ClearColor(Color.Black);

            _shader = new Shader(Environment.CurrentDirectory + $@"/Rendering/Shaders/shader.vert", Environment.CurrentDirectory + $@"/Rendering/Shaders/lighting.frag");

            // Cube instance buffers.
            _cubeVertBufferObject = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, _cubeVertBufferObject);
            GL.BufferData(BufferTarget.ArrayBuffer, _cubeVerts.Length * sizeof(float), _cubeVerts, BufferUsageHint.StaticDraw);

            _cubeVertArrayObject = GL.GenVertexArray();
            GL.BindVertexArray(_cubeVertArrayObject);
            _posAttrib = _shader.GetAttribLocation("aPosition");
            GL.EnableVertexAttribArray(_posAttrib);
            GL.VertexAttribPointer(_posAttrib, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), 0);
            GL.BindVertexArray(0);

            // Normals instance buffers.
            _normVertBufferObject = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, _normVertBufferObject);
            GL.BufferData(BufferTarget.ArrayBuffer, _normalVerts.Length * sizeof(float), _normalVerts, BufferUsageHint.StaticDraw);

            _normVertArrayObject = GL.GenVertexArray();
            GL.BindVertexArray(_normVertArrayObject);
            _normAttrib = _shader.GetAttribLocation("aNormal");
            GL.EnableVertexAttribArray(_normAttrib);
            GL.VertexAttribPointer(_normAttrib, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), 0);
            GL.BindVertexArray(0);

            // Body offset buffers.
            _offsetBufferObject = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, _offsetBufferObject);
            GL.BufferData(BufferTarget.ArrayBuffer, _offsets.Length * Vector4.SizeInBytes, _offsets, BufferUsageHint.StaticDraw);

            _offsetArrayObject = GL.GenVertexArray();
            GL.BindVertexArray(_offsetArrayObject);
            _offsetAttrib = _shader.GetAttribLocation("aOffset");
            GL.EnableVertexAttribArray(_offsetAttrib);
            GL.VertexAttribPointer(_offsetAttrib, 4, VertexAttribPointerType.Float, false, Vector4.SizeInBytes, 0);
            GL.BindVertexArray(0);

            // Body color buffers.
            _colorBufferObject = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, _colorBufferObject);
            GL.BufferData(BufferTarget.ArrayBuffer, _colors.Length * Vector3.SizeInBytes, _colors, BufferUsageHint.StaticDraw);

            _colorArrayObject = GL.GenVertexArray();
            GL.BindVertexArray(_colorArrayObject);
            _colorAttrib = _shader.GetAttribLocation("aObjColor");
            GL.EnableVertexAttribArray(_colorAttrib);
            GL.VertexAttribPointer(_colorAttrib, 3, VertexAttribPointerType.Float, false, Vector3.SizeInBytes, 0);
            GL.BindVertexArray(0);

            // Instance divisors.
            GL.VertexAttribDivisor(_normAttrib, 0);
            GL.VertexAttribDivisor(_posAttrib, 0);
            GL.VertexAttribDivisor(_offsetAttrib, 1);
            GL.VertexAttribDivisor(_colorAttrib, 1);

        }

        public void Render(Body[] bodies, ManualResetEventSlim completeCallback)
        {
            completeCallback.Reset();

            const float time = 0.016f;

            if (!InputHandler.KeyIsDown(Keys.ControlKey))
            {
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
            }


            if (bodies.Length > 0)
            {
                // Render Bodies
                GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
                GL.Enable(EnableCap.DepthTest);
                GL.Enable(EnableCap.LineSmooth);
                GL.Enable(EnableCap.Blend);
                GL.Disable(EnableCap.CullFace);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);

                // var bodies = BodyManager.Bodies;

                this.MakeCurrent();

                //  Draw Body cubes.
                var zOrder = ComputeZOrder(bodies);

                if (_offsets.Length != bodies.Length)
                {
                    _offsets = new Vector4[bodies.Length];
                    GL.BindBuffer(BufferTarget.ArrayBuffer, _offsetBufferObject);
                    GL.BufferData(BufferTarget.ArrayBuffer, _offsets.Length * Vector4.SizeInBytes, _offsets, BufferUsageHint.StaticDraw);

                    _colors = new Vector3[bodies.Length];
                    GL.BindBuffer(BufferTarget.ArrayBuffer, _colorBufferObject);
                    GL.BufferData(BufferTarget.ArrayBuffer, _colors.Length * Vector3.SizeInBytes, _colors, BufferUsageHint.StaticDraw);
                }

                // Update body position offsets and colors via mem map.
                unsafe
                {
                    var offsetPtr = GL.MapNamedBuffer(_offsetArrayObject, BufferAccess.ReadWrite);
                    var offNativePtr = (Vector4*)offsetPtr.ToPointer();

                    var colorPtr = GL.MapNamedBuffer(_colorBufferObject, BufferAccess.ReadWrite);
                    var colorNativePtr = (Vector3*)colorPtr.ToPointer();

                    for (int i = 0; i < bodies.Length; i++)
                    {
                        //  var body = bodies[i];
                        var body = bodies[zOrder[i]];
                        var bPos = body.PositionVec();
                        var bColor = GetStyleColor(body, i);
                        //var bColor = Color.FromArgb(body.Color);
                        var normColor = new Vector3(bColor.R / 255f, bColor.G / 255f, bColor.B / 255f);
                        var offset = new Vector4(bPos, body.Size / 2);

                        offNativePtr[i] = offset;

                        if (body.UID == BodyManager.FollowBodyUID)
                            colorNativePtr[i] = new Vector3(0f, 1.0f, 0f);

                        else
                            colorNativePtr[i] = normColor;

                    }

                    GL.UnmapNamedBuffer(_offsetArrayObject);
                    GL.UnmapNamedBuffer(_colorBufferObject);
                }

                GL.EnableVertexAttribArray(_posAttrib);
                GL.BindBuffer(BufferTarget.ArrayBuffer, _cubeVertBufferObject);
                GL.VertexAttribPointer(_posAttrib, 3, VertexAttribPointerType.Float, false, Vector3.SizeInBytes, 0);

                GL.EnableVertexAttribArray(_colorAttrib);
                GL.BindBuffer(BufferTarget.ArrayBuffer, _colorBufferObject);
                GL.VertexAttribPointer(_colorAttrib, 3, VertexAttribPointerType.Float, false, Vector3.SizeInBytes, 0);

                GL.EnableVertexAttribArray(_offsetAttrib);
                GL.BindBuffer(BufferTarget.ArrayBuffer, _offsetBufferObject);
                GL.VertexAttribPointer(_offsetAttrib, 4, VertexAttribPointerType.Float, false, Vector4.SizeInBytes, 0);

                GL.EnableVertexAttribArray(_normAttrib);
                GL.BindBuffer(BufferTarget.ArrayBuffer, _normVertBufferObject);
                GL.VertexAttribPointer(_normAttrib, 3, VertexAttribPointerType.Float, false, Vector3.SizeInBytes, 0);

                _shader.Use();

                var lightPos = _camera.Position;

                _shader.SetMatrix4("model", Matrix4.Identity);

                if (BodyManager.FollowSelected)
                {
                    var bPos = BodyManager.FollowBody().PositionVec();
                    _shader.SetMatrix4("view", _camera.GetViewMatrix(bPos));
                    lightPos = Vector3.Add(lightPos, bPos);
                }
                else
                {
                    _shader.SetMatrix4("view", _camera.GetViewMatrix());
                }

                _shader.SetMatrix4("projection", _camera.GetProjectionMatrix());
                _shader.SetVector3("lightColor", new Vector3(1.0f, 1.0f, 1.0f));
                _shader.SetVector3("lightPos", lightPos);
                //_shader.SetVector3("viewPos", _camera.Position);
                _shader.SetVector3("viewPos", lightPos);

                _shader.SetFloat("alpha", RenderBase.BodyAlpha / 255f);
                _shader.SetInt("noLight", 0);

                GL.DrawArraysInstanced(PrimitiveType.Quads, 0, _cubeVerts.Length, bodies.Length);

                //  Draw mesh
                if (RenderBase.ShowMesh && BodyManager.Mesh.Length > 1)
                {
                    _shader.SetInt("noLight", 1);

                    var mesh = BodyManager.Mesh;

                    if (_offsets.Length < mesh.Length)
                    {
                        _offsets = new Vector4[mesh.Length];
                        GL.BindBuffer(BufferTarget.ArrayBuffer, _offsetBufferObject);
                        GL.BufferData(BufferTarget.ArrayBuffer, _offsets.Length * Vector4.SizeInBytes, _offsets, BufferUsageHint.StaticDraw);

                        _colors = new Vector3[mesh.Length];
                        GL.BindBuffer(BufferTarget.ArrayBuffer, _colorBufferObject);
                        GL.BufferData(BufferTarget.ArrayBuffer, _colors.Length * Vector3.SizeInBytes, _colors, BufferUsageHint.StaticDraw);
                    }

                    GL.BindBuffer(BufferTarget.ArrayBuffer, _offsetBufferObject);
                    GL.BindBuffer(BufferTarget.ArrayBuffer, _colorBufferObject);

                    unsafe
                    {
                        var offsetPtr = GL.MapNamedBuffer(_offsetArrayObject, BufferAccess.ReadWrite);
                        var offNativePtr = (Vector4*)offsetPtr.ToPointer();

                        var colorPtr = GL.MapNamedBuffer(_colorBufferObject, BufferAccess.ReadWrite);
                        var colorNativePtr = (Vector3*)colorPtr.ToPointer();

                        for (int i = 0; i < mesh.Length; i++)
                        {
                            var cell = mesh[i];
                            var pos = cell.PositionVec();
                            var color = Color.Red;
                            var normColor = new Vector3(color.R / 255f, color.G / 255f, color.B / 255f);
                            var offset = new Vector4(pos, cell.Size / 2);

                            offNativePtr[i] = offset;
                            colorNativePtr[i] = normColor;
                        }

                        GL.UnmapNamedBuffer(_offsetArrayObject);
                        GL.UnmapNamedBuffer(_colorBufferObject);
                    }

                    GL.Disable(EnableCap.CullFace);
                    GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);

                    GL.DrawArraysInstanced(PrimitiveType.Quads, 0, _cubeVerts.Length, mesh.Length);
                }

                GL.BindBuffer(BufferTarget.ArrayBuffer, 0);

                this.SwapBuffers();
            }

            completeCallback.Set();
        }

        public void MoveCameraToCenterMass()
        {
            var cm = BodyManager.CenterOfMass3D();
            _camera.Position = cm;
        }

        private int[] ComputeZOrder(Body[] bodies)
        {
            if (_orderDist.Length != bodies.Length)
            {
                _orderDist = new float[bodies.Length];
                _orderIdx = new int[bodies.Length];
            }

            //for (int i = 0; i < bodies.Length; i++)
            //{
            //    var body = bodies[i];
            //    _orderDist[i] = body.UID;
            //    _orderIdx[i] = i;
            //}

            for (int i = 0; i < bodies.Length; i++)
            {
                var body = bodies[i];
                var pos = body.PositionVec();

                float dist = Vector3.Distance(pos, _camera.Position);
                _orderDist[i] = dist;
                _orderIdx[i] = i;
            }

            Array.Sort(_orderDist, _orderIdx);
            Array.Reverse(_orderIdx);

            return _orderIdx;
        }

        private Color GetStyleColor(Body body, int index)
        {
            Color bodyColor = Color.White;

            switch (RenderBase.DisplayStyle)
            {
                case DisplayStyle.Normal:
                    bodyColor = Color.FromArgb(RenderBase.BodyAlpha, Color.FromArgb(body.Color));
                    GL.ClearColor(Color.Black);

                    break;

                case DisplayStyle.Pressure:
                    bodyColor = RenderBase.GetVariableColor(Color.Blue, Color.Red, Color.Yellow, RenderBase.StyleScaleMax, body.Pressure, true);
                    GL.ClearColor(Color.Black);

                    break;

                case DisplayStyle.Density:
                    bodyColor = RenderBase.GetVariableColor(Color.Blue, Color.Red, Color.Yellow, RenderBase.StyleScaleMax, body.Density / body.Mass, true);
                    GL.ClearColor(Color.Black);

                    break;

                case DisplayStyle.Velocity:
                    bodyColor = RenderBase.GetVariableColor(Color.Blue, Color.Red, Color.Yellow, RenderBase.StyleScaleMax, body.AggregateSpeed(), true);
                    GL.ClearColor(Color.Black);

                    break;

                case DisplayStyle.Index:
                    bodyColor = RenderBase.GetVariableColor(Color.Blue, Color.Red, Color.Yellow, BodyManager.TopUID, body.UID, true);
                    GL.ClearColor(Color.Black);

                    break;

                case DisplayStyle.SpatialOrder:
                    //bodyColor = RenderBase.GetVariableColor(Color.Blue, Color.Red, Color.Yellow, BodyManager.Bodies.Length, index, true);

                    bodyColor = RenderBase.GetVariableColor(Color.Blue, Color.Red, Color.Yellow, BodyManager.Bodies.Length, _orderIdx[index], true);
                    GL.ClearColor(Color.Black);

                    break;

                case DisplayStyle.Force:
                    bodyColor = RenderBase.GetVariableColor(Color.Blue, Color.Red, Color.Yellow, RenderBase.StyleScaleMax, (body.ForceTot / body.Mass), true);
                    GL.ClearColor(Color.Black);

                    break;

                case DisplayStyle.HighContrast:
                    bodyColor = Color.Black;
                    GL.ClearColor(Color.White);

                    break;
            }

            return bodyColor;
        }

        private Tuple<Vector3, Vector3> MouseRay(int x, int y)
        {
            var modelMatrix = _camera.GetViewMatrix();
            var projMatrix = _camera.GetProjectionMatrix();
            var viewport = this.Size;

            var start = new Vector3(x, y, 0.0f).UnProject(projMatrix, modelMatrix, viewport);
            var end = new Vector3(x, y, 1.0f).UnProject(projMatrix, modelMatrix, viewport);
            return new Tuple<Vector3, Vector3>(start, end);
        }

        private bool HitSphere(Body body, Vector3 start, Vector3 end)
        {
            var pos = body.PositionVec();
            var cross = Vector3.Cross(Vector3.Subtract(pos, start), Vector3.Subtract(pos, end)).Length / Vector3.Subtract(end, start).Length;

            if (cross < body.Size / 2f)
                return true;

            return false;
        }

        private void FindClickedBody(Point mousePos)
        {
            var mouseRays = MouseRay(mousePos.X, mousePos.Y);
            bool hitFound = false;

            foreach (var body in BodyManager.Bodies)
            {
                bool hit = HitSphere(body, mouseRays.Item1, mouseRays.Item2);
                if (hit)
                {
                    BodyManager.FollowBodyUID = body.UID;
                    BodyManager.FollowSelected = true;
                    hitFound = true;
                    // Offset camera position to keep selected body in view.
                    _camera.Position = _camera.Position - body.PositionVec();
                    break;
                }
            }

            if (!hitFound)
            {
                // Restore the original position then unselect.
                _camera.Position = _camera.Position + BodyManager.FollowBody().PositionVec();

                BodyManager.FollowBodyUID = -1;
                BodyManager.FollowSelected = false;
            }
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);

            _camera.AspectRatio = this.ClientSize.Width / (float)this.ClientSize.Height;
            GL.Viewport(this.ClientSize);
        }

        protected override void OnKeyUp(KeyEventArgs e)
        {
            InputHandler.KeyUp(e.KeyCode);

            base.OnKeyUp(e);
        }

        protected override void OnKeyDown(KeyEventArgs e)
        {
            InputHandler.KeyDown(e.KeyCode);

            base.OnKeyDown(e);

            if (InputHandler.KeyIsDown(Keys.Q))
            {
                if (cameraSpeed == cameraSpeedFast)
                    cameraSpeed = cameraSpeedSlow;
                else
                    cameraSpeed = cameraSpeedFast;
            }
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            InputHandler.MouseUp(e.Button, e.Location);

            base.OnMouseUp(e);
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            InputHandler.MouseDown(e.Button, e.Location);
            InputHandler.MouseDown(e.Button, _camera.Position);

            base.OnMouseDown(e);

            var mousePos = e.Location;

            if (InputHandler.KeyIsDown(Keys.ControlKey))
                FindClickedBody(e.Location);

            if (e.Button == MouseButtons.Right)
                _lastPos = new Vector2(e.X, e.Y);
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            InputHandler.MouseMove(e.Location);

            base.OnMouseMove(e);

            const float sensitivity = 0.2f;

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

            }
        }

        protected override void OnMouseWheel(MouseEventArgs e)
        {
            InputHandler.MouseWheel(e.Delta);

            base.OnMouseWheel(e);
        }

    }
}
