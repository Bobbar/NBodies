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
            if (FollowSelected)
            {
                if (UIDIndex.ContainsKey(FollowBodyUID))
                {
                    return Bodies[UIDIndex[FollowBodyUID]];
                }
            }

            return new Body();
        }
        public static int UIDToIndex(int uid)
        {
            return UIDIndex[uid];
        }

        public static void CalcDensityAndPressure()
        {
            float GAS_K = 0.1f;
            float FLOAT_EPSILON = 1.192092896e-07f;

            Parallel.For(0, Bodies.Length, a =>
            {

                var bodyA = Bodies[a];

                bodyA.Density = 0;
                bodyA.Pressure = 0;
                // bodyA.Neighbors = 0;

                if (bodyA.InRoche == 1)
                {
                    for (int b = 0; b < Bodies.Length; b++)
                    {
                        var bodyB = Bodies[b];

                        if (bodyB.InRoche == 1)
                        {
                            float DistX = bodyB.LocX - bodyA.LocX;
                            float DistY = bodyB.LocY - bodyA.LocY;
                            float Dist = (DistX * DistX) + (DistY * DistY);
                            float DistSq = (float)Math.Sqrt(Dist);

                            float ksize = bodyA.Size;
                            float ksizeSq = ksize * ksize;
                            // is this distance close enough for kernal/neighbor calcs?
                            if (Dist < ksize)
                            {

                                if (Dist < FLOAT_EPSILON)
                                {
                                    Dist = FLOAT_EPSILON;
                                }

                                // It's a neighbor; accumulate density.
                                float diff = ksizeSq - Dist;
                                double kernRad9 = Math.Pow((double)ksize, 9.0);
                                double factor = (float)(315.0 / (64.0 * Math.PI * kernRad9));

                                double fac = factor * diff * diff * diff;
                                bodyA.Density += (float)(bodyA.Mass * fac);
                            }
                        }
                    }

                    if (bodyA.Density > 0)
                    {
                        bodyA.Pressure = GAS_K * (bodyA.Density);// - DENSITY_OFFSET);
                    }

                    Bodies[a] = bodyA;
                }
            });
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