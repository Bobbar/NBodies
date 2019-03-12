using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ProtoBuf;
using MessagePack;

namespace NBodies.Physics
{

    [ProtoContract]
    [MessagePackObject]
    public struct Body
    {
        [ProtoMember(1)]
        [Key(0)]
        public float PosX;

        [ProtoMember(2)]
        [Key(1)]
        public float PosY;

        [ProtoMember(3)]
        [Key(2)]
        public float Mass;

        [ProtoMember(4)]
        [Key(3)]
        public float VeloX;

        [ProtoMember(5)]
        [Key(4)]
        public float VeloY;

        [ProtoMember(6)]
        [Key(5)]
        public float ForceX;

        [ProtoMember(7)]
        [Key(6)]
        public float ForceY;

        [ProtoMember(8)]
        [Key(7)]
        public float ForceTot;

        [ProtoMember(9)]
        [Key(8)]
        public int Color;

        [ProtoMember(10)]
        [Key(9)]
        public float Size;

        [ProtoMember(11)]
        [Key(10)]
        public int Visible;

        [ProtoMember(12)]
        [Key(11)]
        public int InRoche;

        [ProtoMember(13)]
        [Key(12)]
        public int Flag;

        [ProtoMember(14)]
        [Key(13)]
        public int UID;

        [ProtoMember(15)]
        [Key(14)]
        public float Density;

        [ProtoMember(16)]
        [Key(15)]
        public float Pressure;

        //[IgnoreMember]
        //public int HasCollision;

        [IgnoreMember]
        public int IsExplosion;

        [IgnoreMember]
        public float Lifetime;

        [IgnoreMember]
        public float Age;

        [IgnoreMember]
        public int MeshID;

       // public float Test;

        //[IgnoreMember]
        //public float ElapTime;

        //[IgnoreMember]
        //public float DeltaTime;
    }

}
