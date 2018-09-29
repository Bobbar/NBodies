using NBodies.Rules;
using NBodies.Structures;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using NBodies.CUDA;
namespace NBodies.Rendering
{
    public static class BodyManager
    {
        public static CUDAMain.Body[] Bodies = new CUDAMain.Body[0];

        public static bool FollowSelected = false;

        public static int FollowBodyIndex
        {
            get
            {
                return _followBodyIndex;
            }

            set
            {
                _followBodyIndex = value;

                if (value != -1)
                {
                    FollowBodyUID = Bodies[value].UID;
                }
                else
                {
                    FollowBodyUID = -1;
                }
            }
        }

        public static int FollowBodyUID = -1;

        private static int _followBodyIndex = -1;
        private static List<CUDAMain.Body> _bodyStore = new List<CUDAMain.Body>();
        private static int _currentId = -1;


        public static PointF FollowBodyLoc()
        {
            for (int i = 0; i < Bodies.Length; i++)
            {
                if (Bodies[i].UID == FollowBodyUID)
                {
                    return new PointF((float)Bodies[i].LocX, (float)Bodies[i].LocY);
                }
            }
            return new PointF();
        }

        public static void Move(int index, PointF location)
        {
            Bodies[index].LocX = location.X;
            Bodies[index].LocY = location.Y;
        }

        public static int Add(CUDAMain.Body body)
        {
            _currentId++;

            _bodyStore = Bodies.ToList();
            body.UID = _currentId;
            _bodyStore.Add(body);
            Bodies = _bodyStore.ToArray();

            return _currentId;
        }

        public static void Add(double locX, double locY, double size, double mass, Color color, int blackhole = 0)
        {
            var b = new CUDAMain.Body();

            b.LocX = locX;
            b.LocY = locY;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.SpeedX = 0;
            b.SpeedY = 0;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = 0;
            b.BlackHole = blackhole;
            b.UID = -1;

            Add(b);
        }

        public static void Add(double locX, double locY, double velX, double velY, double size, double mass, Color color)
        {
            var b = new CUDAMain.Body();

            b.LocX = locX;
            b.LocY = locY;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.SpeedX = velX;
            b.SpeedY = velY;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = 0;
            b.BlackHole = 0;
            b.UID = -1;

            Add(b);
        }

        public static void Add(PointF loc, double size, double mass, Color color, int blackhole = 0)
        {
            var b = new CUDAMain.Body();

            b.LocX = loc.X;
            b.LocY = loc.Y;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.SpeedX = 0;
            b.SpeedY = 0;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = 0;
            b.BlackHole = blackhole;
            b.UID = -1;

            Add(b);
        }

        public static int Add(PointF loc, double size, Color color)
        {
            var b = new CUDAMain.Body();

            b.LocX = loc.X;
            b.LocY = loc.Y;
            b.Mass = CalcMass(size);
            b.Size = size;
            b.Color = color.ToArgb();

            b.SpeedX = 0;
            b.SpeedY = 0;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = 0;
            b.BlackHole = 0;
            b.UID = -1;

            return Add(b);
        }

        public static double CalcMass(double size)
        {
            return Math.Sqrt(Math.PI * (Math.Pow(size, 2))) * Matter.Density;
        }
    }
}