using OpenTK;
using OpenTK.Graphics.OpenGL4;
using NBodies.Rendering.Renderables;

namespace NBodies.Rendering.GameObjects
{ 
    public class RenderCharacter : AGameObject
    {
        private float _offset;

        public RenderCharacter(ARenderable model, Vector4 position, float charOffset)
            : base(model, position, Vector4.Zero, Vector4.Zero, 0)
        {
            _offset = charOffset;
           _scale = new Vector3(0f);

        }

        public void SetChar(float charOffset)
        {
            _offset = charOffset;
        }

        public override void Render(Camera camera)
        {
            GL.VertexAttrib2(2, new Vector2(_offset, 0));
            var t2 = Matrix4.CreateTranslation(
                _position.X,
                _position.Y,
                _position.Z);
            var s = Matrix4.CreateScale(_scale);
            var view = Matrix4.LookAt(new Vector3(0), -Vector3.UnitZ, Vector3.UnitY);
            _modelView = s * t2 * view;

            GL.UniformMatrix4(21, false, ref _modelView);
            _model.Render();
        }
    }
}