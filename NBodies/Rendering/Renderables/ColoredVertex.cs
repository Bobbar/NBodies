using OpenTK;
using OpenTK.Graphics;

namespace NBodies.Rendering.Renderables
{
    public struct ColoredVertex
    {
        public const int Size = (4 + 4) * 4; // size of struct in bytes

        private readonly Vector4 _position;
        private readonly Color4 _color;

        public ColoredVertex(Vector4 position, Color4 color)
        {
            _position = position;
            _color = color;
        }
    }


    public struct ColoredVertex2
    {
        public const int Size = (4 + 3) * 4; // size of struct in bytes

        private readonly Vector4 _position;
        private readonly Vector3 _color;

        public ColoredVertex2(Vector4 position, Vector3 color)
        {
            _position = position;
            _color = color;
        }
    }

}