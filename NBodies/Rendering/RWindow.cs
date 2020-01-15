using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Input;
using NBodies.Physics;

namespace NBodies.Rendering
{
    public class RWindow : GameWindow
    {
        private float[] _vertices = new float[0];

        #region Verts
       //        private readonly float[] _vertices =
       //{
       //      14.93127f, -90.61714f, 1.0f,
       //23.95503f, -84.67978f, 1.0f,
       //5.073838f, -64.27119f, 1.0f,
       //18.60629f, -72.3186f, 1.0f,
       //31.68899f, -72.99821f, 1.0f,
       //25.50167f, -76.44541f, 1.0f,
       //19.96881f, -64.79253f, 1.0f,
       //36.26463f, -80.92904f, 1.0f,
       //32.9622f, -78.75596f, 1.0f,
       //51.64996f, -64.94073f, 1.0f,
       //9.765549f, -63.34544f, 1.0f,
       //4.148099f, -52.98913f, 1.0f,
       //13.88581f, -45.97347f, 1.0f,
       //28.9386f, -47.29928f, 1.0f,
       //23.54516f, -34.51681f, 1.0f,
       //39.07346f, -53.44236f, 1.0f,
       //43.10247f, -48.2821f, 1.0f,
       //52.10137f, -56.47321f, 1.0f,
       //47.22218f, -47.78578f, 1.0f,
       //39.13245f, -38.11737f, 1.0f,
       //55.32893f, -43.21667f, 1.0f,
       //7.934533f, -28.79574f, 1.0f,
       //8.519887f, -29.56515f, 1.0f,
       //0.07548749f, -23.33934f, 1.0f,
       //2.734912f, -16.70409f, 1.0f,
       //28.27002f, -28.3868f, 1.0f,
       //20.51974f, -22.31492f, 1.0f,
       //13.33163f, -14.59577f, 1.0f,
       //17.70816f, -0.03702178f, 1.0f,
       //39.61707f, -20.69594f, 1.0f,
       //61.20807f, -17.00258f, 1.0f,
       //37.30641f, -5.038478f, 1.0f,
       //63.61406f, -4.909461f, 1.0f,
       //79.78421f, -58.11053f, 1.0f,
       //71.34524f, -55.51247f, 1.0f,
       //64.6496f, -41.99223f, 1.0f,
       //75.63952f, -41.90312f, 1.0f,
       //64.61807f, -32.63621f, 1.0f,
       //83.74644f, -46.02854f, 1.0f,
       //86.31699f, -37.42536f, 1.0f,
       //94.07162f, -32.13515f, 1.0f,
       //65.97317f, -30.48249f, 1.0f,
       //66.90046f, -31.0794f, 1.0f,
       //90.7562f, -23.43441f, 1.0f,
       //66.55688f, -1.322253f, 1.0f,
       //76.86916f, -2.320189f, 1.0f,
       //-6.004941f, -96.51795f, 1.0f,
       //-4.832317f, -98.1813f, 1.0f,
       //-39.65191f, -89.14269f, 1.0f,
       //-34.18918f, -83.20707f, 1.0f,
       //-58.32391f, -64.69786f, 1.0f,
       //-33.8591f, -64.30525f, 1.0f,
       //-21.75707f, -83.63757f, 1.0f,
       //-22.18953f, -82.69062f, 1.0f,
       //-12.28763f, -93.21712f, 1.0f,
       //-1.065808f, -91.56795f, 1.0f,
       //-12.43814f, -81.75956f, 1.0f,
       //-22.32167f, -74.38197f, 1.0f,
       //-22.94116f, -76.18457f, 1.0f,
       //-74.97587f, -60.73204f, 1.0f,
       //-73.77522f, -60.52835f, 1.0f,
       //-77.42995f, -56.82821f, 1.0f,
       //-69.93311f, -60.91228f, 1.0f,
       //-84.06983f, -40.58578f, 1.0f,
       //-90.93298f, -39.06247f, 1.0f,
       //-70.45675f, -40.44761f, 1.0f,
       //-67.8579f, -41.70257f, 1.0f,
       //-73.22892f, -36.8812f, 1.0f,
       //-72.04869f, -35.24187f, 1.0f,
       //-67.03011f, -33.87349f, 1.0f,
       //-66.7415f, -35.00164f, 1.0f,
       //-75.13609f, -28.05396f, 1.0f,
       //-78.62542f, -31.88453f, 1.0f,
       //-81.45017f, -14.5621f, 1.0f,
       //-86.30823f, -7.730299f, 1.0f,
       //-67.19122f, -3.748873f, 1.0f,
       //-52.08189f, -63.6127f, 1.0f,
       //-55.856f, -51.46439f, 1.0f,
       //-38.00316f, -61.59057f, 1.0f,
       //-36.81921f, -52.03263f, 1.0f,
       //-50.06263f, -43.88168f, 1.0f,
       //-46.26562f, -46.30336f, 1.0f,
       //-45.3181f, -46.11177f, 1.0f,
       //-33.67654f, -35.33005f, 1.0f,
       //-29.60979f, -48.83694f, 1.0f,
       //-17.78353f, -50.89727f, 1.0f,
       //-3.868491f, -56.50233f, 1.0f,
       //-28.85694f, -43.38118f, 1.0f,
       //-11.12123f, -43.73667f, 1.0f,
       //-1.967875f, -36.30467f, 1.0f,
       //-56.96366f, -31.18274f, 1.0f,
       //-58.20228f, -18.27859f, 1.0f,
       //-50.16096f, -23.70933f, 1.0f,
       //-34.22844f, -30.11591f, 1.0f,
       //-33.46315f, -16.13227f, 1.0f,
       //-54.02648f, -8.867299f, 1.0f,
       //-42.05025f, -14.88622f, 1.0f,
       //-47.72266f, -3.656546f, 1.0f,
       //-33.25155f, -2.398944f, 1.0f,
       //-33.30957f, -1.503545f, 1.0f,
       //-35.86562f, -0.5845524f, 1.0f,
       //-20.36843f, -16.31137f, 1.0f,
       //-0.1360968f, -22.43902f, 1.0f,
       //-25.65338f, -15.65294f, 1.0f,
       //-19.21585f, -14.10279f, 1.0f,
       //-2.30886f, -3.246179f, 1.0f,
       //-4.167755f, -4.068399f, 1.0f,
       //2.676365f, 9.085243f, 1.0f,
       //23.06798f, 12.0584f, 1.0f,
       //15.80433f, 28.79826f, 1.0f,
       //47.88871f, 20.95444f, 1.0f,
       //56.97917f, 21.53412f, 1.0f,
       //18.58802f, 34.06402f, 1.0f,
       //31.27433f, 37.94796f, 1.0f,
       //21.11264f, 46.43608f, 1.0f,
       //4.459247f, 55.96493f, 1.0f,
       //3.545098f, 49.85917f, 1.0f,
       //12.64492f, 51.01026f, 1.0f,
       //17.57455f, 52.23544f, 1.0f,
       //18.68552f, 62.94699f, 1.0f,
       //48.11843f, 42.79461f, 1.0f,
       //58.88397f, 40.14213f, 1.0f,
       //67.39391f, 15.38119f, 1.0f,
       //84.08508f, 12.92614f, 1.0f,
       //89.26555f, 15.44382f, 1.0f,
       //76.10937f, 28.83949f, 1.0f,
       //95.72336f, 25.90366f, 1.0f,
       //92.86869f, 31.92682f, 1.0f,
       //96.84442f, 8.430912f, 1.0f,
       //97.02102f, 13.54416f, 1.0f,
       //68.0993f, 45.95199f, 1.0f,
       //88.86682f, 34.77477f, 1.0f,
       //88.05367f, 34.25172f, 1.0f,
       //77.27045f, 50.019f, 1.0f,
       //14.18221f, 71.42853f, 1.0f,
       //5.465597f, 72.47218f, 1.0f,
       //6.536361f, 72.59595f, 1.0f,
       //17.40968f, 66.41393f, 1.0f,
       //17.68701f, 64.46561f, 1.0f,
       //19.10603f, 80.47132f, 1.0f,
       //22.90904f, 82.31332f, 1.0f,
       //44.80869f, 69.39586f, 1.0f,
       //45.06179f, 75.57923f, 1.0f,
       //60.34196f, 69.06531f, 1.0f,
       //44.31354f, 87.73515f, 1.0f,
       //41.57663f, 83.49984f, 1.0f,
       //20.201f, 96.77672f, 1.0f,
       //68.99723f, 70.1066f, 1.0f,
       //-95.22923f, 13.98788f, 1.0f,
       //-74.05875f, 5.995492f, 1.0f,
       //-76.09106f, 6.067307f, 1.0f,
       //-69.63789f, 8.623528f, 1.0f,
       //-70.49886f, 8.229795f, 1.0f,
       //-68.30269f, 28.30487f, 1.0f,
       //-73.25136f, 49.09558f, 1.0f,
       //-58.02701f, 1.776835f, 1.0f,
       //-61.1781f, 10.80411f, 1.0f,
       //-48.85649f, 10.37036f, 1.0f,
       //-53.11782f, 8.62887f, 1.0f,
       //-47.74429f, 0.6538025f, 1.0f,
       //-36.771f, 6.391033f, 1.0f,
       //-36.12581f, 12.89574f, 1.0f,
       //-52.40353f, 25.17707f, 1.0f,
       //-42.31812f, 25.95835f, 1.0f,
       //-37.99136f, 29.00718f, 1.0f,
       //-25.72641f, 7.571894f, 1.0f,
       //-21.14172f, 17.53712f, 1.0f,
       //-17.06151f, 31.75895f, 1.0f,
       //-11.66849f, 22.07329f, 1.0f,
       //-63.43149f, 33.94946f, 1.0f,
       //-49.07629f, 45.06499f, 1.0f,
       //-50.74755f, 45.40047f, 1.0f,
       //-37.84287f, 47.8671f, 1.0f,
       //-63.80585f, 51.01392f, 1.0f,
       //-50.35209f, 49.49234f, 1.0f,
       //-50.57996f, 63.67741f, 1.0f,
       //-43.9809f, 55.50156f, 1.0f,
       //-34.91972f, 51.26352f, 1.0f,
       //-33.88376f, 60.23244f, 1.0f,
       //-32.93352f, 63.14688f, 1.0f,
       //-25.66194f, 38.13636f, 1.0f,
       //-27.27062f, 44.38543f, 1.0f,
       //-19.68454f, 42.78481f, 1.0f,
       //-15.31288f, 36.89322f, 1.0f,
       //-16.48459f, 55.58466f, 1.0f,
       //-16.44306f, 50.71479f, 1.0f,
       //-18.52196f, 53.63592f, 1.0f,
       //-12.12614f, 58.18926f, 1.0f,
       //-14.20423f, 59.93448f, 1.0f,
       //-63.00131f, 77.37866f, 1.0f,
       //-46.92225f, 70.23098f, 1.0f,
       //-43.03994f, 75.80489f, 1.0f,
       //-48.16762f, 81.43594f, 1.0f,
       //-43.00864f, 80.73147f, 1.0f,
       //-34.06583f, 85.7345f, 1.0f,
       //-41.98981f, 89.43711f, 1.0f,
       //-22.7633f, 68.98924f, 1.0f,
       //-24.76095f, 81.62486f, 1.0f,
       //-21.20717f, 86.75637f, 1.0f,
       //-7.082312f, 90.54311f, 1.0f
       //        };
        #endregion Verts

       Camera _camera;

        int _width;
        int _height;

        private int _vertexBufferObject;
        private int _vertexArrayObject;

        private Shader _shader;

        private double _time;
        private bool _firstMove = true;
        private Vector2 _lastPos;

        public RWindow(int width, int height, string title) : base(width, height, GraphicsMode.Default, title) { }

        protected override void OnLoad(EventArgs e)
        {
            GL.ClearColor(0.2f, 0.3f, 0.3f, 1.0f);

            GL.Enable(EnableCap.DepthTest);

            _vertexBufferObject = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vertexBufferObject);
            GL.BufferData(BufferTarget.ArrayBuffer, _vertices.Length * sizeof(float), _vertices, BufferUsageHint.StaticDraw);

            //_elementBufferObject = GL.GenBuffer();
            //GL.BindBuffer(BufferTarget.ElementArrayBuffer, _elementBufferObject);
            //GL.BufferData(BufferTarget.ElementArrayBuffer, _indices.Length * sizeof(uint), _indices, BufferUsageHint.StaticDraw);

            _shader = new Shader(Environment.CurrentDirectory + $@"/Rendering/Shaders/shader.vert", Environment.CurrentDirectory + $@"/Rendering/Shaders/shader.frag");
            _shader.Use();

            //_texture = new Texture("Resources/container.png");
            //_texture.Use();

            //_texture2 = new Texture("Resources/awesomeface.png");
            //_texture2.Use(TextureUnit.Texture1);

            //_shader.SetInt("texture0", 0);
            //_shader.SetInt("texture1", 1);

            //_vertexArrayObject = GL.GenVertexArray();
            //GL.BindVertexArray(_vertexArrayObject);

            //GL.BindBuffer(BufferTarget.ArrayBuffer, _vertexArrayObject);
            //  GL.BindBuffer(BufferTarget.ElementArrayBuffer, _elementBufferObject);


            var vertexLocation = _shader.GetAttribLocation("aPosition");
            GL.EnableVertexAttribArray(vertexLocation);
            GL.VertexAttribPointer(vertexLocation, 3, VertexAttribPointerType.Float, true, 3 * sizeof(float), 0);


            //var texCoordLocation = _shader.GetAttribLocation("aTexCoord");
            //GL.EnableVertexAttribArray(texCoordLocation);
            //GL.VertexAttribPointer(texCoordLocation, 2, VertexAttribPointerType.Float, false, 5 * sizeof(float), 3 * sizeof(float));

            // We initialize the camera so that it is 3 units back from where the rectangle is
            // and give it the proper aspect ratio
            _camera = new Camera(Vector3.UnitZ * 3, Width / (float)Height);

            // We make the mouse cursor invisible so we can have proper FPS-camera movement
           // CursorVisible = false;

            GL.PointSize(5.0f);

            // GL.ClipControl(ClipOrigin.LowerLeft, ClipDepthMode.ZeroToOne);

            base.OnLoad(e);
        }

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            _time += 4.0 * e.Time;

            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            var bodies = BodyManager.Bodies;

            if (bodies.Length == 0)
                return;

            _vertices = new float[bodies.Length * 3];

            for (int i = 0; i < bodies.Length; i++)
            {
                var b = bodies[i];

                _vertices[i * 3] = b.PosX;
                _vertices[(i * 3) + 1] = b.PosY;
                _vertices[(i * 3) + 2] = 0.0f;

            }

           // GL.DeleteBuffer(_vertexBufferObject);

            //_vertexBufferObject = GL.GenBuffer();
            //GL.BindBuffer(BufferTarget.ArrayBuffer, _vertexBufferObject);
            GL.BufferData(BufferTarget.ArrayBuffer, _vertices.Length * sizeof(float), _vertices, BufferUsageHint.StaticDraw);



            // GL.BindVertexArray(_vertexArrayObject);

            //_texture.Use();
            //_texture2.Use(TextureUnit.Texture1);
            _shader.Use();

            var model = Matrix4.Identity;// * Matrix4.CreateRotationX((float)MathHelper.DegreesToRadians(_time));
            _shader.SetMatrix4("model", model);
            _shader.SetMatrix4("view", _camera.GetViewMatrix());
            _shader.SetMatrix4("projection", _camera.GetProjectionMatrix());

            //GL.DrawElements(PrimitiveType.Points, _indices.Length, DrawElementsType.UnsignedInt, 0);
            //  GL.DrawElements(PrimitiveType.Points, _vertices.Length, DrawElementsType.UnsignedInt, 0);
            GL.DrawArrays(PrimitiveType.Points, 0, _vertices.Length);
            SwapBuffers();

            base.OnRenderFrame(e);
        }

        //protected override void OnRenderFrame(FrameEventArgs e)
        //{
        //    _time += 4.0 * e.Time;

        //    GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        //    // GL.BindVertexArray(_vertexArrayObject);

        //    //_texture.Use();
        //    //_texture2.Use(TextureUnit.Texture1);
        //    _shader.Use();

        //    var model = Matrix4.Identity;// * Matrix4.CreateRotationX((float)MathHelper.DegreesToRadians(_time));
        //    _shader.SetMatrix4("model", model);
        //    _shader.SetMatrix4("view", _camera.GetViewMatrix());
        //    _shader.SetMatrix4("projection", _camera.GetProjectionMatrix());

        //    //GL.DrawElements(PrimitiveType.Points, _indices.Length, DrawElementsType.UnsignedInt, 0);
        //    //  GL.DrawElements(PrimitiveType.Points, _vertices.Length, DrawElementsType.UnsignedInt, 0);
        //    GL.DrawArrays(PrimitiveType.Points, 0, _vertices.Length);
        //    SwapBuffers();

        //    base.OnRenderFrame(e);
        //}

        protected override void OnUpdateFrame(FrameEventArgs e)
        {
            if (!Focused) // check to see if the window is focused
            {
                return;
            }

            var input = Keyboard.GetState();

            if (input.IsKeyDown(Key.Escape))
            {
                Exit();
            }

            const float cameraSpeed = 30f;//1.5f;
            const float sensitivity = 0.2f;

            if (input.IsKeyDown(Key.W))
                _camera.Position += _camera.Front * cameraSpeed * (float)e.Time; // Forward 
            if (input.IsKeyDown(Key.S))
                _camera.Position -= _camera.Front * cameraSpeed * (float)e.Time; // Backwards
            if (input.IsKeyDown(Key.A))
                _camera.Position -= _camera.Right * cameraSpeed * (float)e.Time; // Left
            if (input.IsKeyDown(Key.D))
                _camera.Position += _camera.Right * cameraSpeed * (float)e.Time; // Right
            if (input.IsKeyDown(Key.Space))
                _camera.Position += _camera.Up * cameraSpeed * (float)e.Time; // Up 
            if (input.IsKeyDown(Key.LShift))
                _camera.Position -= _camera.Up * cameraSpeed * (float)e.Time; // Down

            // Get the mouse state
            var mouse = Mouse.GetState();

            if (mouse.IsButtonDown(MouseButton.Right))
            {
                if (_firstMove) // this bool variable is initially set to true
                {
                    _lastPos = new Vector2(mouse.X, mouse.Y);
                    _firstMove = false;
                }
                else
                {
                    // Calculate the offset of the mouse position
                    var deltaX = mouse.X - _lastPos.X;
                    var deltaY = mouse.Y - _lastPos.Y;
                    _lastPos = new Vector2(mouse.X, mouse.Y);

                    // Apply the camera pitch and yaw (we clamp the pitch in the camera class)
                    _camera.Yaw += deltaX * sensitivity;
                    _camera.Pitch -= deltaY * sensitivity; // reversed since y-coordinates range from bottom to top
                }

            }


            base.OnUpdateFrame(e);
        }

        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            _camera.Fov -= e.DeltaPrecise;
            // System.Diagnostics.Debug.WriteLine($"{e.DeltaPrecise}  -  {e.Delta}");
            System.Diagnostics.Debug.WriteLine($"{_camera.Fov}");
            base.OnMouseWheel(e);
        }

        protected override void OnResize(EventArgs e)
        {
            GL.Viewport(0, 0, Width, Height);
            // We need to update the aspect ratio once the window has been resized
            _camera.AspectRatio = Width / (float)Height;
            base.OnResize(e);
        }
    }
}
