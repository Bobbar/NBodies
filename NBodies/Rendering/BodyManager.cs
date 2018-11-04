using NBodies.Rules;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using NBodies.Physics;
using System.Threading;
using System.Threading.Tasks;

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

        public static float TotalMass
        {
            get
            {
                return _totalMass;
            }
        }


        private static Dictionary<int, int> UIDIndex = new Dictionary<int, int>();
        private static List<Body> _bodyStore = new List<Body>();
        private static int _currentId = -1;
        private static int _bodyCount = 0;
        private static float _totalMass = 0;

        public static void CullInvisible()
        {
            if (Bodies.Length < 1) return;

            _bodyStore.Clear();
            _bodyStore = Bodies.ToList();
            _bodyStore.RemoveAll((b) => b.Visible == 0);

            Bodies = _bodyStore.ToArray();

            _bodyCount = Bodies.Length;

            _totalMass = 0;
            //_bodyStore.ForEach(b => _totalMass += b.Mass);

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

        public static Body FollowBody()
        {
            try
            {
                if (FollowSelected)
                {
                    if (UIDIndex.ContainsKey(FollowBodyUID))
                    {
                        return Bodies[UIDIndex[FollowBodyUID]];
                    }
                }
            }
            catch
            {
                // Sometimes a race condition occurs, and the key won't be found even though we passed the condition.
                // Fail silently and try again on the next frame.
            }

            return new Body();
        }

        public static int UIDToIndex(int uid)
        {
            if (UIDIndex.ContainsKey(uid))
                return UIDIndex[uid];

            return -1;
        }

        public static Body BodyFromUID(int uid)
        {
            if (UIDIndex.ContainsKey(uid))
            {
                return Bodies[UIDIndex[uid]];
            }

            return new Body();
        }

        public static void Move(int index, PointF location)
        {
            if (index >= 0)
            {
                Bodies[index].LocX = location.X;
                Bodies[index].LocY = location.Y;
            }
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

        //public static void Add(float locX, float locY, float size, float mass, Color color, float viscosity, int blackhole = 0)
        //{
        //    var b = new Body();

        //    b.LocX = locX;
        //    b.LocY = locY;
        //    b.Mass = mass;
        //    b.Size = size;
        //    b.Color = color.ToArgb();

        //    b.SpeedX = 0;
        //    b.SpeedY = 0;
        //    b.ForceX = 0;
        //    b.ForceY = 0;
        //    b.ForceTot = 0;
        //    b.Visible = 1;
        //    b.InRoche = 0;
        //    b.BlackHole = blackhole;
        //    b.Viscosity = viscosity;
        //    b.UID = -1;

        //    Add(b);
        //}

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
            //return (float)Math.Sqrt(Math.PI * (float)(Math.Pow(size, 2))) * Matter.Density;
            return (float)(Math.PI * (float)(Math.Pow(size / 2.0f, 2))) * Matter.Density;

        }

        public static float CalcMass(float size, float density)
        {
            //return (float)Math.Sqrt(Math.PI * (Math.Pow(size, 2))) * density;
            return (float)(Math.PI * (Math.Pow(size / 2.0f, 2))) * density;
        }

        public static float CalcRadius(float area)
        {
            return (float)Math.Sqrt(area / Math.PI);
        }
        public static float AggregateSpeed(this Body body)
        {
            return (float)Math.Sqrt(Math.Pow(body.SpeedX, 2) + Math.Pow(body.SpeedY, 2));
        }

        public static void PrintInfo(this Body body)
        {
            string info = $@"
Index: { Bodies.ToList().IndexOf(body) }
UID: { body.UID }
Mass: { body.Mass }
Size: { body.Size }
InRoche: { body.InRoche }
Density: { body.Density }
Pressure: { body.Pressure }
HasCollision: { body.HasCollision }
Agg. Speed: { body.AggregateSpeed() }
Speed (X,Y): { body.SpeedX }, { body.SpeedY }
Position (X,Y): { body.LocX }, { body.LocY }
Force (X,Y): { body.ForceX }, { body.ForceX }
";

            Console.WriteLine(info);
        }
    }
}