using System.Collections.Generic;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL4;
using NBodies.Rendering.Renderables;

namespace NBodies.Rendering.GameObjects
{
    public class RenderText : AGameObject
    {
        private readonly Vector4 _color;
        public const string Characters = @"qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM0123456789µ§½!""#¤%&/()=?^*@£€${[]}\~¨'-_.:,;<>|°©®±¥";
        private static readonly Dictionary<char, int> Lookup;
        public static readonly float CharacterWidthNormalized;
        // 21x48 per char, 
        public readonly List<RenderCharacter> Text;

        static RenderText()
        {
            Lookup = new Dictionary<char, int>();
            for (int i = 0; i < Characters.Length; i++)
            {
                if (!Lookup.ContainsKey(Characters[i]))
                    Lookup.Add(Characters[i], i);
            }

            CharacterWidthNormalized = (1f / Characters.Length);
        }

        public RenderText(ARenderable model, Vector4 position, Color4 color, string value)
            : base(model, position, Vector4.Zero, Vector4.Zero, 0)
        {
            _color = new Vector4(color.R, color.G, color.B, color.A);
            Text = new List<RenderCharacter>(value.Length);
            _scale = new Vector3(18.0f);

            SetText(value);
        }

        public void SetText(string value)
        {
            Text.Clear();
            for (int i = 0; i < value.Length; i++)
            {
                int offset;
                if (Lookup.TryGetValue(value[i], out offset))
                {
                    var c = new RenderCharacter(Model,
                        new Vector4(_position.X + (i * 11f),
                           _position.Y,
                           _position.Z,
                           _position.W),
                       (offset * CharacterWidthNormalized));

                    c.SetScale(_scale);
                    Text.Add(c);
                }
            }
        }

        public override void Render(Camera camera)
        {
            _model.Bind();
            GL.VertexAttrib4(3, _color);
            for (int i = 0; i < Text.Count; i++)
            {
                var c = Text[i];
                c.Render(camera);
            }
        }
    }
}