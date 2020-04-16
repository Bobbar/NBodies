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
using PixelFormat = OpenTK.Graphics.OpenGL4.PixelFormat;
using NBodies;
using NBodies.Extensions;
using NBodies.Physics;
using NBodies.Rendering;
using NBodies.UI;
using NBodies.UI.KeyActions;
using NBodies.Helpers;
using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using NBodies.Rendering.Renderables;
using NBodies.Rendering.GameObjects;

namespace NBodies.Rendering
{
    public class OpenTKControl : GLControl
    {
        public bool UsePoints
        {
            get { return _usePoints; }

            set { _usePoints = value; }
        }

        private int _cubeVertBufferObject;
        private int _cubePosBufferObject;
        private int _cubesVAO;

        private ColoredVertex2[] _cubePositions = new ColoredVertex2[0];
        private NormalVertex[] _cubeVerts = new NormalVertex[0];

        private int _normAttrib;
        private int _posAttrib;
        private int _positionAttrib;
        private int _colorAttrib;

        private Size _prevSize;
        private Camera _camera;
        private Vector3 _camFollowOffset = new Vector3();

        private Color _clearColor = Color.Black;
        private Color _defaultBodyColor = Color.White;

        private float[] _camSpeeds = new float[]
        {
            10f,
            50f,
            200f,
            600f,
            1000f,
            3000f,
            10000f,
            30000f
        };

        private float cameraSpeed = 200f;

        private bool _firstMove = true;
        private Vector2 _lastPos;

        private Shader _shader;
        private Shader _textShader;

        private float[] _orderDist = new float[0];
        private int[] _orderIdx = new int[0];

        private RenderText _text;

        private int _pointTex;
        private string[] _pointSpriteTextures;
        private int _currentPointTextIdx = 0;
        private bool _usePoints = false;
        private bool _useShaderSpheres = true;

        private int _newBodyId = -1;


        private Stopwatch _timer = new Stopwatch();

        private bool _ready = false;

        // Bloom buffers and shaders.
        private int _hdrFBO;
        private int[] _colorBuffers = new int[2];
        private int _rboDepth;
        private DrawBuffersEnum[] _attachments;
        private int[] _pingpongFBO = new int[2];
        private int[] _pingpongColorbuffers = new int[2];
        private Shader _blurShader;
        private Shader _bloomFinalShader;
        private int _quadVAO = 0;
        private int _quadVBO = -1;

        public OpenTKControl(GraphicsMode mode) : base(mode)
        {
        }

        public void Init()
        {
            _prevSize = this.ClientSize;
            _camera = new Camera(Vector3.UnitZ, this.ClientSize.Width / (float)this.ClientSize.Height);

            this.MakeCurrent();

            GL.ClearColor(Color.Black);

            _shader = new Shader(Environment.CurrentDirectory + $@"/Rendering/Shaders/shaderVert.vert", Environment.CurrentDirectory + $@"/Rendering/Shaders/lightingFrag.frag");
            _textShader = new Shader(Environment.CurrentDirectory + $@"/Rendering/Shaders/textVert.vert", Environment.CurrentDirectory + $@"/Rendering/Shaders/textFrag.frag");
            _blurShader = new Shader(Environment.CurrentDirectory + $@"/Rendering/Shaders/blurVert.vert", Environment.CurrentDirectory + $@"/Rendering/Shaders/blurFrag.frag");
            _bloomFinalShader = new Shader(Environment.CurrentDirectory + $@"/Rendering/Shaders/bloomFinalVert.vert", Environment.CurrentDirectory + $@"/Rendering/Shaders/bloomFinalFrag.frag");

            var textModel = new TexturedRenderObject(RenderObjectFactory.CreateTexturedCharacter(), _textShader.Handle, @"Rendering\Textures\font singleline.bmp");
            _text = new RenderText(textModel, new Vector4(0), Color.LimeGreen, "");

            _cubeVerts = RenderObjectFactory.CreateQuadCubeNormal();
            _cubesVAO = GL.GenVertexArray();
            GL.BindVertexArray(_cubesVAO);

            // Cube instance buffer.
            _cubeVertBufferObject = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, _cubeVertBufferObject);
            GL.BufferData(BufferTarget.ArrayBuffer, _cubeVerts.Length * NormalVertex.Size, _cubeVerts, BufferUsageHint.StaticDraw);

            _posAttrib = _shader.GetAttribLocation("cubeVert");
            GL.VertexAttribPointer(_posAttrib, 3, VertexAttribPointerType.Float, false, NormalVertex.Size, 0);
            GL.VertexAttribDivisor(_posAttrib, 0);
            GL.EnableVertexAttribArray(_posAttrib);

            _normAttrib = _shader.GetAttribLocation("cubeNormal");
            GL.VertexAttribPointer(_normAttrib, 3, VertexAttribPointerType.Float, false, NormalVertex.Size, Vector3.SizeInBytes);
            GL.VertexAttribDivisor(_normAttrib, 0);
            GL.EnableVertexAttribArray(_normAttrib);


            // Body position buffer.
            _cubePosBufferObject = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, _cubePosBufferObject);
            GL.BufferData(BufferTarget.ArrayBuffer, _cubePositions.Length * ColoredVertex2.Size, _cubePositions, BufferUsageHint.StaticDraw);

            _positionAttrib = _shader.GetAttribLocation("aPosition");
            GL.VertexAttribPointer(_positionAttrib, 4, VertexAttribPointerType.Float, false, ColoredVertex2.Size, 0);
            GL.EnableVertexAttribArray(_positionAttrib);

            _colorAttrib = _shader.GetAttribLocation("aObjColor");
            GL.VertexAttribPointer(_colorAttrib, 3, VertexAttribPointerType.Float, false, ColoredVertex2.Size, Vector4.SizeInBytes);
            GL.EnableVertexAttribArray(_colorAttrib);

            GL.BindVertexArray(0);


            _pointSpriteTextures = new string[]
            {
                "", // Use shader spheres.
                @"Rendering\Textures\circle_fuzzy.png",
                @"Rendering\Textures\circle.png",
                @"Rendering\Textures\cloud.png",
                @"Rendering\Textures\cloud_med.png",
                @"Rendering\Textures\cloud_dense.png",
                @"Rendering\Textures\bubble.png",
                @"Rendering\Textures\star.png"
            };


            //_pointTex = InitTextures($@"Rendering\Textures\bubble.png");
            //_pointTex = InitTextures($@"Rendering\Textures\circle.png");
            _pointTex = InitTextures($@"Rendering\Textures\circle_fuzzy.png");
            //_pointTex = InitTextures($@"Rendering\Textures\cloud_dense.png");
            // _pointTex = InitTextures($@"Rendering\Textures\cloud_med.png");
            //_pointTex = InitTextures($@"Rendering\Textures\cloud.png");
            //_pointTex = InitTextures($@"Rendering\Textures\star.png");


            InitBloomBuffers();

            _blurShader.Use();
            _blurShader.SetInt("blurTex", 0);

            _bloomFinalShader.Use();
            _bloomFinalShader.SetInt("scene", 0);
            _bloomFinalShader.SetInt("bloomBlur", 1);

            GL.Enable(EnableCap.DepthTest);

            _ready = true;
        }

        private void InitBloomBuffers()
        {
            //GL.GenBuffers(1, out _hdrFBO);
            _hdrFBO = GL.GenFramebuffer();
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, _hdrFBO);

            GL.GenTextures(2, _colorBuffers);
            for (int i = 0; i < 2; i++)
            {
                GL.BindTexture(TextureTarget.Texture2D, _colorBuffers[i]);
                // GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgb16f, ClientSize.Width, ClientSize.Height, 0, PixelFormat.Rgb, PixelType.Float, IntPtr.Zero);
                GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, ClientSize.Width, ClientSize.Height, 0, PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);

                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMinFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

                GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0 + i, TextureTarget.Texture2D, _colorBuffers[i], 0);
            }

            GL.GenRenderbuffers(1, out _rboDepth);
            GL.BindRenderbuffer(RenderbufferTarget.Renderbuffer, _rboDepth);
            GL.RenderbufferStorage(RenderbufferTarget.Renderbuffer, RenderbufferStorage.DepthComponent, ClientSize.Width, ClientSize.Height);
            GL.FramebufferRenderbuffer(FramebufferTarget.Framebuffer, FramebufferAttachment.DepthAttachment, RenderbufferTarget.Renderbuffer, _rboDepth);

            _attachments = new DrawBuffersEnum[]
            {
                DrawBuffersEnum.ColorAttachment0,
                DrawBuffersEnum.ColorAttachment1
            };
            GL.DrawBuffers(2, _attachments);

            var check = GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer);
            if (check != FramebufferErrorCode.FramebufferComplete)
                Debugger.Break();



            GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

            GL.GenFramebuffers(2, _pingpongFBO);
            GL.GenTextures(2, _pingpongColorbuffers);

            for (int i = 0; i < 2; i++)
            {
                GL.BindFramebuffer(FramebufferTarget.Framebuffer, _pingpongFBO[i]);
                GL.BindTexture(TextureTarget.Texture2D, _pingpongColorbuffers[i]);
                // GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgb16f, ClientSize.Width, ClientSize.Height, 0, PixelFormat.Rgb, PixelType.Float, IntPtr.Zero);
                //GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, ClientSize.Width, ClientSize.Height, 0, PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
                //GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, ClientSize.Width / 2, ClientSize.Height / 2, 0, PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
                GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, ClientSize.Width / 2, ClientSize.Height / 2, 0, PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);

                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMinFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
                GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, _pingpongColorbuffers[i], 0);

                if (GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != FramebufferErrorCode.FramebufferComplete)
                    Debugger.Break();
            }
        }

        private void DeleteBloomBuffers()
        {
            GL.DeleteFramebuffer(_hdrFBO);
            GL.DeleteTextures(2, _colorBuffers);
            GL.DeleteRenderbuffer(_rboDepth);
            GL.DeleteFramebuffers(2, _pingpongFBO);
            GL.DeleteTextures(2, _pingpongColorbuffers);
        }

        private void ResizeBuffers()
        {
            if (_prevSize != this.ClientSize)
            {
                DeleteBloomBuffers();
                InitBloomBuffers();
                _prevSize = this.ClientSize;
            }
        }

        public void Render(Body[] bodies, ManualResetEventSlim completeCallback)
        {
            completeCallback.Reset();

            const float time = 0.016f;

            if (!InputHandler.KeyIsDown(Keys.ControlKey))
            {
                Vector3 camPos = _camera.Position;

                if (BodyManager.FollowSelected)
                    camPos = _camFollowOffset;

                if (InputHandler.KeyIsDown(Keys.W))
                    camPos += _camera.Front * cameraSpeed * time; // Forward 
                if (InputHandler.KeyIsDown(Keys.S))
                    camPos -= _camera.Front * cameraSpeed * time; // Backwards
                if (InputHandler.KeyIsDown(Keys.A))
                    camPos -= _camera.Right * cameraSpeed * time; // Left
                if (InputHandler.KeyIsDown(Keys.D))
                    camPos += _camera.Right * cameraSpeed * time; // Right
                if (InputHandler.KeyIsDown(Keys.Space))
                    camPos += _camera.Up * cameraSpeed * time; // Up 
                if (InputHandler.KeyIsDown(Keys.ShiftKey))
                    camPos -= _camera.Up * cameraSpeed * time; // Down

                if (BodyManager.FollowSelected)
                    _camFollowOffset = camPos;
                else
                    _camera.Position = camPos;
            }

            CheckPointTex();
            ResizeBuffers();

            if (_camera.HasMoved)
            {
                var pos = new Vector3(this.Size.Width / 2f, this.Size.Height / 2f, 0.1f).UnProject(_camera.GetProjectionMatrix(), _camera.GetViewMatrix(), this.Size);
                var dir = pos - _camera.Position;
                ViewportHelpers.CameraDirection = dir;
            }

            if (bodies.Length > 0)
            {
                // Render Bodies
                GL.ClearColor(_clearColor);
                GL.Enable(EnableCap.Blend);
                GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

                if (_usePoints)
                {
                    GL.Enable(EnableCap.PointSprite);
                    GL.Enable(EnableCap.VertexProgramPointSize);
                    GL.Enable(EnableCap.Texture2D);

                    GL.Disable(EnableCap.DepthTest);

                    if (_useShaderSpheres)
                    {
                        GL.Enable(EnableCap.DepthTest);
                        GL.DepthMask(true);

                        if (RenderVars.BodyAlpha == 255)
                            GL.Disable(EnableCap.Blend);
                    }
                    else
                    {
                        GL.ActiveTexture(TextureUnit.Texture0);
                        GL.BindTexture(TextureTarget.Texture2D, _pointTex);
                        _shader.SetInt("spriteTex", 0);
                    }
                }
                else
                {
                    GL.Enable(EnableCap.DepthTest);
                    GL.Enable(EnableCap.LineSmooth);
                    GL.Enable(EnableCap.Blend);

                    GL.Disable(EnableCap.CullFace);
                    GL.Disable(EnableCap.PointSprite);
                    GL.Disable(EnableCap.VertexProgramPointSize);
                    GL.Disable(EnableCap.Texture2D);

                    GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
                }

                if (RenderVars.BloomEnabled)
                {
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, _hdrFBO);
                    GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
                }
                else
                {
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
                    GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
                }


                _shader.Use();

                GL.BindVertexArray(_cubesVAO);

                if (_usePoints)
                {
                    GL.VertexAttribDivisor(_positionAttrib, 0);
                    GL.VertexAttribDivisor(_colorAttrib, 0);
                    GL.EnableVertexAttribArray(_positionAttrib);
                    GL.EnableVertexAttribArray(_colorAttrib);

                    if (_useShaderSpheres)
                        _shader.SetInt("usePoint", 2);
                    else
                        _shader.SetInt("usePoint", 1);
                }
                else
                {
                    GL.VertexAttribDivisor(_positionAttrib, 1);
                    GL.VertexAttribDivisor(_colorAttrib, 1);
                    GL.EnableVertexAttribArray(_positionAttrib);
                    GL.EnableVertexAttribArray(_colorAttrib);
                    _shader.SetInt("usePoint", 0);
                }

                // Offset camera position for follow mode.
                if (BodyManager.FollowSelected)
                {
                    var bPos = BodyManager.FollowBody().PositionVec();
                    _camera.Position = bPos + _camFollowOffset;
                }

                ViewportHelpers.CameraPos = _camera.Position;

                _shader.SetMatrix4("model", Matrix4.Identity);
                _shader.SetMatrix4("view", _camera.GetViewMatrix());
                _shader.SetMatrix4("projection", _camera.GetProjectionMatrix());

                _shader.SetVector3("lightColor", new Vector3(1.0f, 1.0f, 1.0f));
                _shader.SetVector3("lightPos", _camera.Position);
                _shader.SetVector3("viewPos", _camera.Position);

                _shader.SetFloat("alpha", RenderVars.BodyAlpha / 255f);
                _shader.SetInt("noLight", 0);

                // For correct point sprite scaling.
                float nearPlaneHeight = (float)Math.Abs(this.ClientSize.Width - this.ClientSize.Height) / (2 * (float)Math.Tan(0.5 * _camera.Fov * Math.PI / 180.0));
                _shader.SetFloat("nearPlaneHeight", nearPlaneHeight);

                // Don't draw bodies if alpha is 0.
                if (RenderVars.BodyAlpha > 0)
                {
                    // Update body positions and colors.
                    // Compute Z-order for correct blending.
                    var zOrder = ComputeZOrder(bodies);

                    // Realloc if needed.
                    if (_cubePositions.Length < bodies.Length)
                        _cubePositions = new ColoredVertex2[bodies.Length];
            
                    // Set positions, colors and sizes.
                    for (int i = 0; i < bodies.Length; i++)
                    {
                        //  var body = bodies[i];
                        var body = bodies[zOrder[i]];
                        var bPos = body.PositionVec();
                      
                        // Position and size.
                        var cubePos = new Vector4(bPos, body.Size / 2);

                        // Green for following body. Otherwise use style color.
                        if (body.UID == BodyManager.FollowBodyUID)
                        {
                            _cubePositions[i] = new ColoredVertex2(cubePos, new Vector3(0f, 1.0f, 0f));
                        }
                        else
                        {
                            var bColor = GetStyleColor(body, i);
                            var normColor = new Vector3(bColor.R / 255f, bColor.G / 255f, bColor.B / 255f);
                            _cubePositions[i] = new ColoredVertex2(cubePos, normColor);
                        }
                    }

                    GL.BindBuffer(BufferTarget.ArrayBuffer, _cubePosBufferObject);
                    GL.BufferData(BufferTarget.ArrayBuffer, _cubePositions.Length * ColoredVertex2.Size, IntPtr.Zero, BufferUsageHint.StaticDraw);
                    GL.BufferSubData<ColoredVertex2>(BufferTarget.ArrayBuffer, IntPtr.Zero, _cubePositions.Length * ColoredVertex2.Size, _cubePositions);

                    // Draw.
                    if (_usePoints)
                        GL.DrawArrays(PrimitiveType.Points, 0, bodies.Length);
                    else
                        GL.DrawArraysInstanced(PrimitiveType.Quads, 0, _cubeVerts.Length, bodies.Length);
                }

                // Draw mesh if needed.
                DrawMesh();

                GL.BindVertexArray(0);

                if (RenderVars.BloomEnabled)
                {
                    int width = ClientSize.Width;
                    int height = ClientSize.Height;
                    int fac = 2;
                    int dsWidth = (width + fac - 1) / fac;
                    int dsHeight = (height + fac - 1) / fac;


                    // Do 2 pass gaussian blur to bloom buffer.
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

                    _blurShader.Use();
                    _blurShader.SetInt("copy", 1);
                    GL.Viewport(0, 0, ClientSize.Width / 2, ClientSize.Height / 2);
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, _pingpongFBO[0]);
                    GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
                    GL.ActiveTexture(TextureUnit.Texture0);
                    GL.BindTexture(TextureTarget.Texture2D, _colorBuffers[1]);
                    _blurShader.SetInt("blurTex", 0);
                    RenderFullScreenQuad();

                    _blurShader.SetInt("copy", 0);
                    _blurShader.SetInt("horizontal", 1);
                    _blurShader.SetVector4("offset", new Vector4(1.0f / dsWidth, 1.0f / dsHeight, 0.5f / dsWidth, 0.5f / dsHeight));
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, _pingpongFBO[1]);
                    GL.ActiveTexture(TextureUnit.Texture0);
                    GL.BindTexture(TextureTarget.Texture2D, _pingpongColorbuffers[0]);
                    _blurShader.SetInt("blurTex", 0);
                    RenderFullScreenQuad();

                    _blurShader.SetInt("copy", 0);
                    _blurShader.SetInt("horizontal", 0);
                    _blurShader.SetVector4("offset", new Vector4(1.0f / dsWidth, 1.0f / dsHeight, 0.5f / dsWidth, 0.5f / dsHeight));
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, _pingpongFBO[0]);
                    GL.ActiveTexture(TextureUnit.Texture0);
                    GL.BindTexture(TextureTarget.Texture2D, _pingpongColorbuffers[1]);
                    _blurShader.SetInt("blurTex", 0);
                    RenderFullScreenQuad();



                    //float off = RenderVars.Blur;
                    //_blurShader.SetInt("copy", 0);
                    //_blurShader.SetVector2("offset", new Vector2(off / ClientSize.Width, 0));
                    //GL.BindFramebuffer(FramebufferTarget.Framebuffer, _pingpongFBO[1]);
                    //GL.ActiveTexture(TextureUnit.Texture0);
                    //GL.BindTexture(TextureTarget.Texture2D, _pingpongColorbuffers[0]);
                    //_blurShader.SetInt("blurTex", 0);
                    //RenderFullScreenQuad();

                    //_blurShader.SetVector2("offset", new Vector2(0, off / ClientSize.Height));
                    //GL.BindFramebuffer(FramebufferTarget.Framebuffer, _pingpongFBO[0]);
                    //GL.ActiveTexture(TextureUnit.Texture0);
                    //GL.BindTexture(TextureTarget.Texture2D, _pingpongColorbuffers[1]);
                    //_blurShader.SetInt("blurTex", 0);
                    //RenderFullScreenQuad();

                    //off += 2.4f;
                    //_blurShader.SetVector2("offset", new Vector2(off / ClientSize.Width, 0));
                    //GL.BindFramebuffer(FramebufferTarget.Framebuffer, _pingpongFBO[1]);
                    //GL.ActiveTexture(TextureUnit.Texture0);
                    //GL.BindTexture(TextureTarget.Texture2D, _pingpongColorbuffers[0]);
                    //_blurShader.SetInt("blurTex", 0);
                    //RenderFullScreenQuad();

                    //_blurShader.SetVector2("offset", new Vector2(0, off / ClientSize.Height));
                    //GL.BindFramebuffer(FramebufferTarget.Framebuffer, _pingpongFBO[0]);
                    //GL.ActiveTexture(TextureUnit.Texture0);
                    //GL.BindTexture(TextureTarget.Texture2D, _pingpongColorbuffers[1]);
                    //_blurShader.SetInt("blurTex", 0);
                    //RenderFullScreenQuad();


                    // Blend bloom and original buffers to produce final image.
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

                    GL.Viewport(0, 0, ClientSize.Width, ClientSize.Height);
                    GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

                    _bloomFinalShader.Use();

                    // OG image sample.
                    GL.ActiveTexture(TextureUnit.Texture0);
                    GL.BindTexture(TextureTarget.Texture2D, _colorBuffers[0]);
                    _bloomFinalShader.SetInt("scene", 0);

                    // Blurred bloom sample.
                    GL.ActiveTexture(TextureUnit.Texture1);
                    GL.BindTexture(TextureTarget.Texture2D, _pingpongColorbuffers[0]);
                    _bloomFinalShader.SetInt("bloomBlur", 1);

                    //
                    _bloomFinalShader.SetInt("bloom", Convert.ToInt32(RenderVars.BloomEnabled));
                    _bloomFinalShader.SetFloat("gamma", RenderVars.Gamma);
                    _bloomFinalShader.SetFloat("exposure", RenderVars.Exposure);
                    RenderFullScreenQuad();
                }


                // Draw stats and overlay text.
                DrawStatsAndOverlays();

                GL.Finish();
                this.SwapBuffers();
            }

            completeCallback.Set();
        }

        private void RenderFullScreenQuad()
        {
            if (_quadVAO == 0)
            {
                float[] quadVerts = new float[]
                {
                     // positions        // texture Coords
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f
                };

                //_quadVAO = GL.GenVertexArray();
                //_quadVBO = GL.GenBuffer();
                GL.GenVertexArrays(1, out _quadVAO);
                GL.GenBuffers(1, out _quadVBO);
                GL.BindVertexArray(_quadVAO);
                GL.BindBuffer(BufferTarget.ArrayBuffer, _quadVBO);
                GL.BufferData(BufferTarget.ArrayBuffer, quadVerts.Length * 4, quadVerts, BufferUsageHint.StaticDraw);
                GL.EnableVertexAttribArray(0);
                GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 5 * 4, 0);
                GL.VertexAttribDivisor(0, 0);

                GL.EnableVertexAttribArray(1);
                GL.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, 5 * 4, 3 * 4);
                GL.VertexAttribDivisor(1, 0);

            }

            GL.BindVertexArray(_quadVAO);
            GL.DrawArrays(PrimitiveType.TriangleStrip, 0, 4);
            GL.BindVertexArray(0);
        }

        private void DrawStatsAndOverlays()
        {
            var elapTime = TimeSpan.FromSeconds(MainLoop.TotalTime * 10000);

            var stats = new List<string>();

            stats.Add($@"FPS: {MainLoop.CurrentFPS}  ({Math.Round(MainLoop.PeakFPS, 2)})");
            stats.Add($@"Count: {MainLoop.FrameCount}");
            stats.Add($@"Time: {elapTime.Days} days {elapTime.Hours} hr {elapTime.Minutes} min");
            stats.Add($@"Bodies: {BodyManager.BodyCount}");
            stats.Add($@"Cell Size: {Math.Pow(2, MainLoop.CellSizeExp)}");
            stats.Add($@"Mesh Levels: {MainLoop.MeshLevels}");

            if (OpenCLPhysics.NNUsingBrute)
            {
                stats.Add($@"NN Search: Brute");

            }
            else
            {
                stats.Add($@"NN Search: Grid");
                stats.Add($@"Grid Passes: {OpenCLPhysics.GridPasses}");
            }

            if (BodyManager.FollowSelected)
            {
                var body = BodyManager.FollowBody();

                stats.Add($@"Density: {body.Density}");
                stats.Add($@"Press: {body.Pressure}");
                stats.Add($@"Agg. Speed: {body.AggregateSpeed()}");
            }

            if (MainLoop.Recorder.RecordingActive)
            {
                stats.Add($@"Rec Size (MB): {Math.Round((MainLoop.RecordedSize() / (float)1000000), 2)}");
            }

            GL.Enable(EnableCap.DepthTest);
            GL.Enable(EnableCap.LineSmooth);
            GL.Enable(EnableCap.Blend);
            GL.Enable(EnableCap.Texture2D);
            GL.Disable(EnableCap.CullFace);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);

            _textShader.Use();

            var proj = Matrix4.CreateOrthographicOffCenter(0, ClientSize.Width, 0, ClientSize.Height, -100f, 400f);
            GL.UniformMatrix4(20, false, ref proj);

            float lineHeight = 20f;
            float yPos = ClientSize.Height - 20f;

            foreach (var stat in stats)
            {
                _text.SetPosition(new Vector4(20f, yPos, 0, 1));
                _text.SetText(stat);
                yPos -= lineHeight;
                _text.Render(_camera);
            }

            // Draw overlays.
            Vector2 center = new Vector2((ClientSize.Width / 2) - 100f, ClientSize.Height - 20f);

            foreach (var overlay in RenderVars.OverLays)
            {
                if (overlay.Visible)
                {
                    _text.SetPosition(new Vector4(center.X, center.Y, 0, 1));
                    _text.SetText(overlay.Value);
                    _text.Render(_camera);
                }
            }
        }

        public void DrawMesh()
        {
            //  Draw mesh
            if (RenderVars.ShowMesh && BodyManager.Mesh.Length > 1)
            {
                GL.VertexAttribDivisor(_positionAttrib, 1);
                GL.VertexAttribDivisor(_colorAttrib, 1);
                GL.EnableVertexAttribArray(_positionAttrib);
                GL.EnableVertexAttribArray(_colorAttrib);

                _shader.SetInt("usePoint", 0);
                _shader.SetFloat("alpha", 1.0f);
                _shader.SetInt("noLight", 1);

                GL.Enable(EnableCap.DepthTest);
                GL.Enable(EnableCap.LineSmooth);
                GL.Enable(EnableCap.Blend);

                GL.Disable(EnableCap.CullFace);
                GL.Disable(EnableCap.PointSprite);
                GL.Disable(EnableCap.VertexProgramPointSize);

                var mesh = BodyManager.Mesh;

                if (_cubePositions.Length < mesh.Length)
                {
                    _cubePositions = new ColoredVertex2[mesh.Length];
                    GL.BindBuffer(BufferTarget.ArrayBuffer, _cubePosBufferObject);
                    GL.BufferData(BufferTarget.ArrayBuffer, _cubePositions.Length * ColoredVertex2.Size, _cubePositions, BufferUsageHint.StaticDraw);
                }

                unsafe
                {
                    var cubePtr = GL.MapNamedBuffer(_cubePosBufferObject, BufferAccess.WriteOnly);
                    var cubeNativePtr = (ColoredVertex2*)cubePtr.ToPointer();

                    for (int i = 0; i < mesh.Length; i++)
                    {
                        var cell = mesh[i];
                        var pos = cell.PositionVec();
                        var color = Color.Red;
                        var normColor = new Vector3(color.R / 255f, color.G / 255f, color.B / 255f);
                        var cubePos = new Vector4(pos, cell.Size / 2);

                        cubeNativePtr[i] = new ColoredVertex2(cubePos, normColor);
                    }

                    GL.UnmapNamedBuffer(_cubePosBufferObject);
                }

                GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);

                GL.DrawArraysInstanced(PrimitiveType.Quads, 0, _cubeVerts.Length, mesh.Length);

                GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
            }
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
            Color bodyColor = _defaultBodyColor;

            switch (RenderVars.DisplayStyle)
            {
                case DisplayStyle.Normal:
                    bodyColor = Color.FromArgb(RenderVars.BodyAlpha, Color.FromArgb(body.Color));
                    _clearColor = Color.Black;

                    break;

                case DisplayStyle.Pressure:
                    bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, RenderVars.StyleScaleMax, body.Pressure, true);
                    _clearColor = Color.Black;

                    break;

                case DisplayStyle.Density:
                    bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, RenderVars.StyleScaleMax, body.Density / body.Mass, true);
                    _clearColor = Color.Black;

                    break;

                case DisplayStyle.Velocity:
                    bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, RenderVars.StyleScaleMax, body.AggregateSpeed(), true);
                    _clearColor = Color.Black;

                    break;

                case DisplayStyle.Index:
                    bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, BodyManager.TopUID, body.UID, true);
                    _clearColor = Color.Black;

                    break;

                case DisplayStyle.SpatialOrder:
                    //bodyColor = RenderBase.GetVariableColor(Color.Blue, Color.Red, Color.Yellow, BodyManager.Bodies.Length, index, true);

                    bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, BodyManager.Bodies.Length, _orderIdx[index], true);
                    _clearColor = Color.Black;

                    break;

                case DisplayStyle.Force:
                    bodyColor = GetVariableColor(Color.Blue, Color.Red, Color.Yellow, RenderVars.StyleScaleMax, (body.ForceTot / body.Mass), true);
                    _clearColor = Color.Black;

                    break;

                case DisplayStyle.HighContrast:
                    bodyColor = Color.Black;
                    _clearColor = Color.White;

                    break;
            }

            return bodyColor;
        }

        public static Color GetVariableColor(Color startColor, Color midColor, Color endColor, float maxValue, float currentValue, bool translucent = false)
        {
            const int maxIntensity = 255;
            float intensity = 0;
            long r1 = 0;
            long g1 = 0;
            long b1 = 0;
            long r2 = 0;
            long g2 = 0;
            long b2 = 0;

            if (currentValue <= (maxValue / 2f))
            {
                r1 = startColor.R;
                g1 = startColor.G;
                b1 = startColor.B;

                r2 = midColor.R;
                g2 = midColor.G;
                b2 = midColor.B;

                maxValue = maxValue / 2f;
            }
            else
            {
                r1 = midColor.R;
                g1 = midColor.G;
                b1 = midColor.B;

                r2 = endColor.R;
                g2 = endColor.G;
                b2 = endColor.B;

                maxValue = maxValue / 2f;
                currentValue = currentValue - maxValue;
            }

            if (currentValue > 0)
            {
                // Compute the intensity of the end color.
                intensity = (maxIntensity / (maxValue / currentValue));
            }

            // Clamp the intensity within the max.
            if (intensity > maxIntensity) intensity = maxIntensity;

            // Calculate the new RGB values from the intensity.
            int newR, newG, newB;
            newR = (int)(r1 + (r2 - r1) / (float)maxIntensity * intensity);
            newG = (int)(g1 + (g2 - g1) / (float)maxIntensity * intensity);
            newB = (int)(b1 + (b2 - b1) / (float)maxIntensity * intensity);

            if (translucent)
            {
                return Color.FromArgb(RenderVars.BodyAlpha, newR, newG, newB);
            }
            else
            {
                return Color.FromArgb(newR, newG, newB);
            }
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
                    _camFollowOffset = _camera.Position;
                    break;
                }
            }

            if (!hitFound)
            {
                // Restore the original position then unselect.
                // _camera.Position = _camera.Position + BodyManager.FollowBody().PositionVec();

                BodyManager.FollowBodyUID = -1;
                BodyManager.FollowSelected = false;
            }
        }

        private void CheckPointTex()
        {
            var idx = RenderVars.PointSpriteTexIdx;
            if (idx >= 0 && idx < _pointSpriteTextures.Length)
            {
                if (_currentPointTextIdx != idx)
                {
                    if (idx == 0)
                    {
                        _useShaderSpheres = true;
                    }
                    else
                    {
                        _useShaderSpheres = false;
                        GL.DeleteTexture(_pointTex);
                        _pointTex = InitTextures(_pointSpriteTextures[idx]);
                    }

                    _currentPointTextIdx = idx;
                }
            }
            else
            {
                RenderVars.PointSpriteTexIdx = _currentPointTextIdx;
            }
        }

        private int InitTextures(string filename)
        {
            int texture;

            texture = GL.GenTexture();

            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, texture);

            using (var image = new Bitmap(filename))
            {
                // First, we get our pixels from the bitmap we loaded.
                // Arguments:
                //   The pixel area we want. Typically, you want to leave it as (0,0) to (width,height), but you can
                //   use other rectangles to get segments of textures, useful for things such as spritesheets.
                //   The locking mode. Basically, how you want to use the pixels. Since we're passing them to OpenGL,
                //   we only need ReadOnly.
                //   Next is the pixel format we want our pixels to be in. In this case, ARGB will suffice.
                //   We have to fully qualify the name because OpenTK also has an enum named PixelFormat.
                var data = image.LockBits(
                    new Rectangle(0, 0, image.Width, image.Height),
                    ImageLockMode.ReadOnly,
                    System.Drawing.Imaging.PixelFormat.Format32bppArgb);

                // Now that our pixels are prepared, it's time to generate a texture. We do this with GL.TexImage2D
                // Arguments:
                //   The type of texture we're generating. There are various different types of textures, but the only one we need right now is Texture2D.
                //   Level of detail. We can use this to start from a smaller mipmap (if we want), but we don't need to do that, so leave it at 0.
                //   Target format of the pixels. This is the format OpenGL will store our image with.
                //   Width of the image
                //   Height of the image.
                //   Border of the image. This must always be 0; it's a legacy parameter that Khronos never got rid of.
                //   The format of the pixels, explained above. Since we loaded the pixels as ARGB earlier, we need to use BGRA.
                //   Data type of the pixels.
                //   And finally, the actual pixels.
                GL.TexImage2D(TextureTarget.Texture2D,
                    0,
                    PixelInternalFormat.Rgba,
                    image.Width,
                    image.Height,
                    0,
                    PixelFormat.Bgra,
                    PixelType.UnsignedByte,
                    data.Scan0);
            }


            // your image will fail to render at all (usually resulting in pure black instead).
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);


            // Now, set the wrapping mode. S is for the X axis, and T is for the Y axis.
            // We set this to Repeat so that textures will repeat when wrapped. Not demonstrated here since the texture coordinates exactly match
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);

            // Next, generate mipmaps.
            // Mipmaps are smaller copies of the texture, scaled down. Each mipmap level is half the size of the previous one
            // Generated mipmaps go all the way down to just one pixel.
            // OpenGL will automatically switch between mipmaps when an object gets sufficiently far away.
            // This prevents distant objects from having their colors become muddy, as well as saving on memory.
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);

            GL.BindTexture(TextureTarget.Texture2D, 0);


            return texture;
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);

            if (_ready)
            {
                _camera.AspectRatio = this.ClientSize.Width / (float)this.ClientSize.Height;
                GL.Viewport(this.ClientSize);
            }
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

            const int oneKey = 49;
            const int eightKey = 56;

            int spdIdx = 0;
            for (int i = oneKey; i <= eightKey; i++)
            {
                var key = (Keys)i;
                var spd = _camSpeeds[spdIdx++];

                if (InputHandler.KeyIsDown(key))
                    cameraSpeed = spd;
            }

            if (InputHandler.KeyIsDown(Keys.N))
            {
                if (_newBodyId == -1)
                {
                    MainLoop.WaitForPause();

                    var pos = new Vector3(this.Size.Width / 2f, this.Size.Height / 2f, 0.1f).UnProject(_camera.GetProjectionMatrix(), _camera.GetViewMatrix(), this.Size);
                    var velo = pos - _camera.Position;

                    var nb = BodyManager.NewBody(pos, velo, 5, BodyManager.CalcMass(5), ColorHelper.RandomColor(), 1);
                    BodyManager.Add(nb, false);

                    _newBodyId = nb.UID;

                    Debug.WriteLine($@"NewID: {_newBodyId}");
                }
                else
                {
                    var nb = BodyManager.BodyFromUID(_newBodyId);

                    MainLoop.ResumePhysics(true);

                    _newBodyId = -1;
                }



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

            const float sensitivity = 0.13f;

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

            if (!InputHandler.KeysDown)
            {
                if (e.Delta > 0)
                    _camera.Fov -= 1;
                else
                    _camera.Fov -= -1;
            }

            if (_newBodyId != -1)
            {
                if (InputHandler.KeyIsDown(Keys.M))
                {
                    var nb = BodyManager.BodyFromUID(_newBodyId);
                    nb.Size += e.Delta > 0 ? 1 : -1;
                    nb.Mass = BodyManager.CalcMass(nb.Size);
                    Debug.WriteLine($@"Size: {nb.Size}  Mass: {nb.Mass}");
                    BodyManager.Bodies[BodyManager.UIDToIndex(_newBodyId)] = nb;

                }

                if (InputHandler.KeyIsDown(Keys.V))
                {
                    var nb = BodyManager.BodyFromUID(_newBodyId);
                    var velo = nb.VelocityVec();
                    var len = velo.Length;
                    var newLen = len + (e.Delta * 0.2f);
                    velo.X *= newLen / len;
                    velo.Y *= newLen / len;
                    velo.Z *= newLen / len;


                    nb.VeloX = velo.X;
                    nb.VeloY = velo.Y;
                    nb.VeloZ = velo.Z;
                    Debug.WriteLine($@"Velo: {velo.ToString()}");

                    BodyManager.Bodies[BodyManager.UIDToIndex(_newBodyId)] = nb;
                }
            }
        }

    }
}
