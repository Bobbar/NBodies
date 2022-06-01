using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cloo;

namespace NBodies.Physics
{
    public class MeshGpuBuffers
    {
        public ComputeBuffer<int2> Indexes;
        public ComputeBuffer<int2> NeighborBounds;
        public ComputeBuffer<int2> BodyBounds;
        public ComputeBuffer<int2> ChildBounds;
        public ComputeBuffer<float4> CenterMass;
        public ComputeBuffer<int4> SizeParentLevel;
        public ComputeBuffer<int> Neighbors;
    }

    public class MeshHostBuffers
    {
        public int2[] Indexes = new int2[1];
        public int2[] NeighborBounds = new int2[1];
        public int2[] BodyBounds = new int2[1];
        public int2[] ChildBounds = new int2[1];
        public float4[] CenterMass = new float4[1];
        public int4[] SizeParentLevel = new int4[1];
    }
}
