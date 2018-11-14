using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using ProtoBuf;

namespace NBodies.Physics
{

    [Cudafy(eCudafyType.Struct)]
    [ProtoContract]
    public struct Body
    {
        [ProtoMember(1)]
        public float LocX;

        [ProtoMember(2)]
        public float LocY;

        [ProtoMember(3)]
        public float Mass;

        [ProtoMember(4)]
        public float SpeedX;

        [ProtoMember(5)]
        public float SpeedY;

        [ProtoMember(6)]
        public float ForceX;

        [ProtoMember(7)]
        public float ForceY;

        [ProtoMember(8)]
        public float ForceTot;

        [ProtoMember(9)]
        public int Color;

        [ProtoMember(10)]
        public float Size;

        [ProtoMember(11)]
        public int Visible;

        [ProtoMember(12)]
        public int InRoche;

        [ProtoMember(13)]
        public int BlackHole;

        [ProtoMember(14)]
        public int UID;

        [ProtoMember(15)]
        public float Density;

        [ProtoMember(16)]
        public float Pressure;

        public int HasCollision;

        public int IsExplosion;

        public float Lifetime;

        public float Age;

        public float ElapTime;

        public float DeltaTime;
    }
   
}
