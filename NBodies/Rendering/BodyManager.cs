using NBodies.Rules;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using NBodies.Physics;

namespace NBodies.Rendering
{
    public static class BodyManager
    {
        public static Body[] Bodies = new Body[0];
        public static bool FollowSelected = false;
        public static int FollowBodyUID = -1;

        public static int BodyCount
        {
            get
            {
                return _bodyCount;
            }
        }

        private static Dictionary<int, int> UIDIndex = new Dictionary<int, int>();
        private static List<Body> _bodyStore = new List<Body>();
        private static int _currentId = -1;
        private static int _bodyCount = 0;

        public static void CullInvisible()
        {
            if (Bodies.Length < 1) return;

            _bodyStore.Clear();
            _bodyStore = Bodies.ToList();
            _bodyStore.RemoveAll((b) => b.Visible == 0);

            Bodies = _bodyStore.ToArray();

            _bodyCount = Bodies.Length;

            RebuildUIDIndex();
        }

        public static void ClearBodies()
        {
            FollowSelected = false;
            FollowBodyUID = -1;
            Bodies = new Body[0];
            UIDIndex.Clear();
            _bodyStore.Clear();
            _currentId = -1;
            _bodyCount = 0;
        }

        public static void ReplaceBodies(Body[] bodies)
        {
            ClearBodies();

            _currentId = bodies.Max((b) => b.UID);

            Bodies = bodies;

            RebuildUIDIndex();
        }

        public static void RebuildUIDIndex()
        {
            UIDIndex.Clear();

            for (int i = 0; i < Bodies.Length; i++)
            {
                UIDIndex.Add(Bodies[i].UID, i);
            }
        }

        public static PointF FollowBodyLoc()
        {
            if (UIDIndex.ContainsKey(FollowBodyUID))
                return new PointF(Bodies[UIDToIndex(FollowBodyUID)].LocX, Bodies[UIDToIndex(FollowBodyUID)].LocY);
            return new PointF();
        }

        public static int UIDToIndex(int uid)
        {
            return UIDIndex[uid];
        }

        public static void Move(int index, PointF location)
        {
            Bodies[index].LocX = location.X;
            Bodies[index].LocY = location.Y;
        }

        public static int Add(Body body)
        {
            _currentId++;

            _bodyStore = Bodies.ToList();

            body.UID = _currentId;

            _bodyStore.Add(body);
            Bodies = _bodyStore.ToArray();

            UIDIndex.Add(_currentId, Bodies.Length - 1);

            return _currentId;
        }

        public static void Add(float locX, float locY, float size, float mass, Color color, int blackhole = 0)
        {
            var b = new Body();

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

        public static void Add(float locX, float locY, float velX, float velY, float size, float mass, Color color)
        {
            var b = new Body();

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

        public static void Add(float locX, float locY, float velX, float velY, float size, float mass, Color color, int inRoche)
        {
            var b = new Body();

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
            b.InRoche = inRoche;
            b.BlackHole = 0;
            b.UID = -1;

            Add(b);
        }

        public static void Add(PointF loc, float size, float mass, Color color, int blackhole = 0)
        {
            var b = new Body();

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

        public static int Add(PointF loc, float size, Color color)
        {
            var b = new Body();

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

        public static float CalcMass(float size)
        {
            return (float)Math.Sqrt(Math.PI * (float)(Math.Pow(size, 2))) * Matter.Density;
        }

        public static float CalcMass(float size, float density)
        {
            return (float)Math.Sqrt(Math.PI * (Math.Pow(size, 2))) * density;
        }

        public static float CalcRadius(float area)
        {
            return (float)Math.Sqrt(area / Math.PI);
        }
    }
}