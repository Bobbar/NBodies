using OpenTK;
using OpenTK.Graphics;
using NBodies.Rendering.GameObjects;
using NBodies.Rendering.Renderables;
using System.Collections.Generic;

namespace NBodies.Rendering
{
    public class RenderObjectFactory
    {
        public static TexturedVertex[] CreateTexturedCharacter()
        {
            float h = 1;
            float w = RenderText.CharacterWidthNormalized;
            float side = 1f / 2f; // half side - and other half

            TexturedVertex[] vertices =
            {
                new TexturedVertex(new Vector4(-side, -side, side, 1.0f),    new Vector2(0, h)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),     new Vector2(w, h)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),     new Vector2(0, 0)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),     new Vector2(0, 0)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),     new Vector2(w, h)),
                new TexturedVertex(new Vector4(side, side, side, 1.0f),      new Vector2(w, 0)),
            };
            return vertices;
        }

        public static NormalVertex[] CreateQuadCubeNormal()
        {
            float[] _cubeVerts =
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
            float[] _normalVerts =
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

            var verts = new List<NormalVertex>();

            for (int i = 0; i < _cubeVerts.Length; i += 3)
            {
                Vector3 p = new Vector3(_cubeVerts[i], _cubeVerts[i + 1], _cubeVerts[i + 2]);
                Vector3 n = new Vector3(_normalVerts[i], _normalVerts[i + 1], _normalVerts[i + 2]);

                verts.Add(new NormalVertex(p, n));
            }

            return verts.ToArray();
        }


        public static ColoredVertex[] CreateSolidCube(float side, Color4 color)
        {
            side = side / 2f; // half side - and other half
            ColoredVertex[] vertices =
            {
                new ColoredVertex(new Vector4(-side, -side, -side, 1.0f),   color),
                new ColoredVertex(new Vector4(-side, -side, side, 1.0f),    color),
                new ColoredVertex(new Vector4(-side, side, -side, 1.0f),    color),
                new ColoredVertex(new Vector4(-side, side, -side, 1.0f),    color),
                new ColoredVertex(new Vector4(-side, -side, side, 1.0f),    color),
                new ColoredVertex(new Vector4(-side, side, side, 1.0f),     color),

                new ColoredVertex(new Vector4(side, -side, -side, 1.0f),    color),
                new ColoredVertex(new Vector4(side, side, -side, 1.0f),     color),
                new ColoredVertex(new Vector4(side, -side, side, 1.0f),     color),
                new ColoredVertex(new Vector4(side, -side, side, 1.0f),     color),
                new ColoredVertex(new Vector4(side, side, -side, 1.0f),     color),
                new ColoredVertex(new Vector4(side, side, side, 1.0f),      color),

                new ColoredVertex(new Vector4(-side, -side, -side, 1.0f),   color),
                new ColoredVertex(new Vector4(side, -side, -side, 1.0f),    color),
                new ColoredVertex(new Vector4(-side, -side, side, 1.0f),    color),
                new ColoredVertex(new Vector4(-side, -side, side, 1.0f),    color),
                new ColoredVertex(new Vector4(side, -side, -side, 1.0f),    color),
                new ColoredVertex(new Vector4(side, -side, side, 1.0f),     color),

                new ColoredVertex(new Vector4(-side, side, -side, 1.0f),    color),
                new ColoredVertex(new Vector4(-side, side, side, 1.0f),     color),
                new ColoredVertex(new Vector4(side, side, -side, 1.0f),     color),
                new ColoredVertex(new Vector4(side, side, -side, 1.0f),     color),
                new ColoredVertex(new Vector4(-side, side, side, 1.0f),     color),
                new ColoredVertex(new Vector4(side, side, side, 1.0f),      color),

                new ColoredVertex(new Vector4(-side, -side, -side, 1.0f),   color),
                new ColoredVertex(new Vector4(-side, side, -side, 1.0f),    color),
                new ColoredVertex(new Vector4(side, -side, -side, 1.0f),    color),
                new ColoredVertex(new Vector4(side, -side, -side, 1.0f),    color),
                new ColoredVertex(new Vector4(-side, side, -side, 1.0f),    color),
                new ColoredVertex(new Vector4(side, side, -side, 1.0f),     color),

                new ColoredVertex(new Vector4(-side, -side, side, 1.0f),    color),
                new ColoredVertex(new Vector4(side, -side, side, 1.0f),     color),
                new ColoredVertex(new Vector4(-side, side, side, 1.0f),     color),
                new ColoredVertex(new Vector4(-side, side, side, 1.0f),     color),
                new ColoredVertex(new Vector4(side, -side, side, 1.0f),     color),
                new ColoredVertex(new Vector4(side, side, side, 1.0f),      color),
            };
            return vertices;
        }

        public static TexturedVertex[] CreateTexturedCube(float side, float textureWidth, float textureHeight)
        {
            float h = textureHeight;
            float w = textureWidth;
            side = side / 2f; // half side - and other half

            TexturedVertex[] vertices =
            {
                new TexturedVertex(new Vector4(-side, -side, -side, 1.0f),   new Vector2(0, h)),
                new TexturedVertex(new Vector4(-side, -side, side, 1.0f),    new Vector2(w, h)),
                new TexturedVertex(new Vector4(-side, side, -side, 1.0f),    new Vector2(0, 0)),
                new TexturedVertex(new Vector4(-side, side, -side, 1.0f),    new Vector2(0, 0)),
                new TexturedVertex(new Vector4(-side, -side, side, 1.0f),    new Vector2(w, h)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),     new Vector2(w, 0)),

                new TexturedVertex(new Vector4(side, -side, -side, 1.0f),    new Vector2(w, 0)),
                new TexturedVertex(new Vector4(side, side, -side, 1.0f),     new Vector2(0, 0)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),     new Vector2(w, h)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),     new Vector2(w, h)),
                new TexturedVertex(new Vector4(side, side, -side, 1.0f),     new Vector2(0, 0)),
                new TexturedVertex(new Vector4(side, side, side, 1.0f),      new Vector2(0, h)),

                new TexturedVertex(new Vector4(-side, -side, -side, 1.0f),   new Vector2(w, 0)),
                new TexturedVertex(new Vector4(side, -side, -side, 1.0f),    new Vector2(0, 0)),
                new TexturedVertex(new Vector4(-side, -side, side, 1.0f),    new Vector2(w, h)),
                new TexturedVertex(new Vector4(-side, -side, side, 1.0f),    new Vector2(w, h)),
                new TexturedVertex(new Vector4(side, -side, -side, 1.0f),    new Vector2(0, 0)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),     new Vector2(0, h)),

                new TexturedVertex(new Vector4(-side, side, -side, 1.0f),    new Vector2(w, 0)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),     new Vector2(0, 0)),
                new TexturedVertex(new Vector4(side, side, -side, 1.0f),     new Vector2(w, h)),
                new TexturedVertex(new Vector4(side, side, -side, 1.0f),     new Vector2(w, h)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),     new Vector2(0, 0)),
                new TexturedVertex(new Vector4(side, side, side, 1.0f),      new Vector2(0, h)),

                new TexturedVertex(new Vector4(-side, -side, -side, 1.0f),   new Vector2(0, h)),
                new TexturedVertex(new Vector4(-side, side, -side, 1.0f),    new Vector2(w, h)),
                new TexturedVertex(new Vector4(side, -side, -side, 1.0f),    new Vector2(0, 0)),
                new TexturedVertex(new Vector4(side, -side, -side, 1.0f),    new Vector2(0, 0)),
                new TexturedVertex(new Vector4(-side, side, -side, 1.0f),    new Vector2(w, h)),
                new TexturedVertex(new Vector4(side, side, -side, 1.0f),     new Vector2(w, 0)),

                new TexturedVertex(new Vector4(-side, -side, side, 1.0f),    new Vector2(0, h)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),     new Vector2(w, h)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),     new Vector2(0, 0)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),     new Vector2(0, 0)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),     new Vector2(w, h)),
                new TexturedVertex(new Vector4(side, side, side, 1.0f),      new Vector2(w, 0)),
            };
            return vertices;
        }

        public static TexturedVertex[] CreateTexturedCube6(float side, float textureWidth, float textureHeight)
        {
            float h = textureHeight;
            side = side / 2f; // half side - and other half
            var tx10 = 0f;
            var tx11 = textureWidth / 6f;

            var tx20 = textureWidth / 6f;
            var tx21 = (textureWidth / 6f) * 2f;

            var tx30 = (textureWidth / 6f) * 2f;
            var tx31 = (textureWidth / 6f) * 3f;

            var tx40 = (textureWidth / 6f) * 3f;
            var tx41 = (textureWidth / 6f) * 4f;

            var tx50 = (textureWidth / 6f) * 4f;
            var tx51 = (textureWidth / 6f) * 5f;

            var tx60 = (textureWidth / 6f) * 5f;
            var tx61 = (textureWidth / 6f) * 6f;

            TexturedVertex[] vertices =
            {
                new TexturedVertex(new Vector4(-side, -side, -side, 1.0f),  new Vector2(tx10, h)),
                new TexturedVertex(new Vector4(-side, -side, side, 1.0f),   new Vector2(tx11, h)),
                new TexturedVertex(new Vector4(-side, side, -side, 1.0f),   new Vector2(tx10, 0)),
                new TexturedVertex(new Vector4(-side, side, -side, 1.0f),   new Vector2(tx10, 0)),
                new TexturedVertex(new Vector4(-side, -side, side, 1.0f),   new Vector2(tx11, h)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),    new Vector2(tx11, 0)),

                new TexturedVertex(new Vector4(side, -side, -side, 1.0f),   new Vector2(tx21, 0)),
                new TexturedVertex(new Vector4(side, side, -side, 1.0f),    new Vector2(tx20, 0)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),    new Vector2(tx21, h)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),    new Vector2(tx21, h)),
                new TexturedVertex(new Vector4(side, side, -side, 1.0f),    new Vector2(tx20, 0)),
                new TexturedVertex(new Vector4(side, side, side, 1.0f),     new Vector2(tx20, h)),

                new TexturedVertex(new Vector4(-side, -side, -side, 1.0f),  new Vector2(tx31, 0)),
                new TexturedVertex(new Vector4(side, -side, -side, 1.0f),   new Vector2(tx30, 0)),
                new TexturedVertex(new Vector4(-side, -side, side, 1.0f),   new Vector2(tx31, h)),
                new TexturedVertex(new Vector4(-side, -side, side, 1.0f),   new Vector2(tx31, h)),
                new TexturedVertex(new Vector4(side, -side, -side, 1.0f),   new Vector2(tx30, 0)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),    new Vector2(tx30, h)),

                new TexturedVertex(new Vector4(-side, side, -side, 1.0f),   new Vector2(tx41, 0)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),    new Vector2(tx40, 0)),
                new TexturedVertex(new Vector4(side, side, -side, 1.0f),    new Vector2(tx41, h)),
                new TexturedVertex(new Vector4(side, side, -side, 1.0f),    new Vector2(tx41, h)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),    new Vector2(tx40, 0)),
                new TexturedVertex(new Vector4(side, side, side, 1.0f),     new Vector2(tx40, h)),

                new TexturedVertex(new Vector4(-side, -side, -side, 1.0f),  new Vector2(tx50, h)),
                new TexturedVertex(new Vector4(-side, side, -side, 1.0f),   new Vector2(tx51, h)),
                new TexturedVertex(new Vector4(side, -side, -side, 1.0f),   new Vector2(tx50, 0)),
                new TexturedVertex(new Vector4(side, -side, -side, 1.0f),   new Vector2(tx50, 0)),
                new TexturedVertex(new Vector4(-side, side, -side, 1.0f),   new Vector2(tx51, h)),
                new TexturedVertex(new Vector4(side, side, -side, 1.0f),    new Vector2(tx51, 0)),

                new TexturedVertex(new Vector4(-side, -side, side, 1.0f),   new Vector2(tx60, h)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),    new Vector2(tx61, h)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),    new Vector2(tx60, 0)),
                new TexturedVertex(new Vector4(-side, side, side, 1.0f),    new Vector2(tx60, 0)),
                new TexturedVertex(new Vector4(side, -side, side, 1.0f),    new Vector2(tx61, h)),
                new TexturedVertex(new Vector4(side, side, side, 1.0f),     new Vector2(tx61, 0)),

            };
            return vertices;
        }
    }
}