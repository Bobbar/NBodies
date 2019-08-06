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
        public float ForceTot
        {
            get
            {
                return (float)Math.Sqrt((Math.Pow(ForceX, 2) + Math.Pow(ForceY, 2)));
            }

            set { var dummy = value; }

        }

        [ProtoMember(9)]
        [Key(8)]
        public int Color;

        [ProtoMember(10)]
        [Key(9)]
        public float Size;

        /// <summary>
        /// Visible and Culled are flipped values of the same flag.
        /// This is done to maintain backwards compatibility with saved states.
        /// 
        /// Visible = 1 (true) == Culled = 0 (false)
        /// </summary>
        [ProtoMember(11)]
        [Key(10)]
        public int Visible
        {
            get { return Convert.ToInt32(!HasFlag(Flags.Culled)); }

            set
            {
                SetFlag(Flags.Culled, !Convert.ToBoolean(value));
            }
        }

        [ProtoMember(12)]
        [Key(11)]
        public int InRoche
        {
            get { return Convert.ToInt32(HasFlag(Flags.InRoche)); }

            set
            {
                SetFlag(Flags.InRoche, Convert.ToBoolean(value));
            }
        }


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

        [IgnoreMember]
        public float Lifetime;

        [IgnoreMember]
        public int MeshID;


       
        [IgnoreMember]
        public bool Culled
        {
            get { return HasFlag(Flags.Culled); }

            set
            {
                SetFlag(Flags.Culled, value);
            }
        }

        [IgnoreMember]
        public bool IsExplosion
        {
            get { return HasFlag(Flags.IsExplosion); }

            set
            {
                SetFlag(Flags.IsExplosion, value);

                // Explosion particle should always be in roche.
                if (value)
                    SetFlag(Flags.InRoche, true);
            }
        }

        [IgnoreMember]
        public bool IsBlackHole
        {
            get
            {
                return HasFlag(Flags.BlackHole);
            }

            set
            {
                SetFlag(Flags.BlackHole, value);
            }
        }

        //public Body(int dummy) : this()
        //{
        //    Flag = 1;
        //    UID = -1;
        //    MeshID = -1;


        //    //IsExplosion = false;
        //    //IsBlackHole = false;
        //    IsExplosion = 0;
        //    IsBlackHole = 0;
        //    PosX = 0.0f;
        //    PosY = 0.0f;
        //    Mass = 0.0f;
        //    Size = 0.0f;
        //    Color = 0;
        //    VeloX = 0.0f;
        //    VeloY = 0.0f;
        //    ForceX = 0.0f;
        //    ForceY = 0.0f;
        //    ForceTot = 0.0f;
        //    Density = 0.0f;
        //    Pressure = 0.0f;
        //    Culled = false;
        //    InRoche = 0;
        //    Lifetime = -100;

        //}


        public void SetFlag(Flags flag, bool enabled)
        {
            if (enabled)
                SetFlag((int)flag);
            else
                RemoveFlag((int)flag);
        }

        public void SetFlag(Flags flag)
        {
            SetFlag((int)flag);
        }

        public void SetFlag(int flag)
        {
            if (!HasFlag(flag))
                Flag += flag;
        }

        public bool HasFlag(Flags flag)
        {
            return HasFlag((int)flag);
        }

        public bool HasFlag(int flag)
        {
            return (Flag & flag) != 0;
        }

        public void RemoveFlag(int flag)
        {
            if (HasFlag(flag))
                Flag -= flag;
        }
    }
}
