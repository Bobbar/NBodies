using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public class BufferDims
    {
        public string Name;

        public int Capacity;
        public int Size;

        public float GrowFactor = 2.0f;//2.0f;//1.4f;
        public float ShrinkFactor = 4f;

        public bool ExactSize = false;

        public BufferDims(string name, int capacity, int size)
        {
            Name = name;
            Capacity = capacity;
            Size = size;
        }

        public BufferDims(string name, int capacity, int size, bool exactSize)
        {
            Name = name;
            Capacity = capacity;
            Size = size;
            ExactSize = exactSize;
        }


    }
}
