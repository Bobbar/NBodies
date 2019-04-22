using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cloo;

namespace NBodies.Physics
{
    public class BufferDims
    {
        public string Name;
        public long HandleVal;
        public int Capacity;
        public int Size;
        public float GrowFactor = 1.4f;//2.0f;
        public float ShrinkFactor = 4f;
        public bool ExactSize = false;

        public BufferDims(string name, int capacity, int size, bool exactSize)
        {
            Name = name;
            Capacity = capacity;
            Size = size;
            ExactSize = exactSize;
        }

        public BufferDims(long handleVal, int capacity, int size, bool exactSize)
        {
            Name = handleVal.ToString();
            HandleVal = handleVal;
            Capacity = capacity;
            Size = size;
            ExactSize = exactSize;
        }


    }
}
