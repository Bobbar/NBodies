using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK;
using OpenTK.Graphics;


namespace NBodies.Rendering.Renderables
{
    public struct NormalVertex
    {
        public const int Size = (3 + 3) * 4; // size of struct in bytes

        private readonly Vector3 _position;
        private readonly Vector3 _normal;

        public NormalVertex(Vector3 position, Vector3 normal)
        {
            _position = position;
            _normal = normal;
        }


    }
}
