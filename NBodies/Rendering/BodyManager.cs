using NBodies.Physics;
using NBodies.Rules;
using NBodies.Shapes;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace NBodies.Rendering
{
    public static class BodyManager
    {
        public static Body[] Bodies = new Body[0];
        public static MeshPoint[] Mesh = new MeshPoint[0];
        public static int[,] MeshBodies = new int[0, 0];
        public static bool FollowSelected = false;
        public static int FollowBodyUID = -1;

        public static int BodyCount
        {
            get
            {
                return _bodyCount;
            }
        }

        public static double TotalMass
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
        private static double _totalMass = 0;

        private static System.Diagnostics.Stopwatch timer = new System.Diagnostics.Stopwatch();

        public static void CullInvisible()
        {
            if (Bodies.Length < 1) return;

            _bodyStore.Clear();
            _bodyStore = Bodies.ToList();
            _bodyStore.RemoveAll((b) => b.Visible == 0);

            _bodyStore.RemoveAll((b) => b.Age > b.Lifetime);

            Bodies = _bodyStore.ToArray();

            _bodyCount = Bodies.Length;

            _totalMass = 0;
            //_bodyStore.ForEach(b => _totalMass += b.Mass);

            RebuildUIDIndex();

            //   CheckSetForNextDT();
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

        public static void BuildMesh()
        {
            var maxX = Bodies.Max(b => b.LocX);
            var minX = Bodies.Min(b => b.LocX);

            var maxY = Bodies.Max(b => b.LocY);
            var minY = Bodies.Min(b => b.LocY);

            float nodeSize = 0;
            float nodeRad = 0;
            int nodes = 20; //100;
            float meshSize = 0;

            var wX = maxX - minX;
            var wY = maxY - minY;

            if (wX > wY)
            {
                meshSize = wX;
            }
            else
            {
                meshSize = wY;
            }

            nodeSize = meshSize / nodes;
            nodeRad = nodeSize / 2f;

            MeshPoint[] mesh = new MeshPoint[nodes * nodes];

            List<MeshPoint> meshList = new List<MeshPoint>();

            //  int[][] meshBodies = new int[nodes * nodes][];
            //    int[,] meshBodies = new int[nodes * nodes,0];


            float curX = minX;
            float curY = maxY;

            for (int i = 0; i < nodes * nodes; i++)
            {
                mesh[i].LocX = curX + (nodeRad);
                mesh[i].LocY = curY - (nodeRad);
                mesh[i].Mass = 0f;
                mesh[i].Count = 0;


                if (curX + nodeSize <= maxX)
                {
                    curX += nodeSize;
                }
                else
                {
                    if (curY - nodeSize >= minY)
                    {
                        curX = minX;
                        curY -= nodeSize;
                    }
                }
            }

            List<int[]> meshBods = new List<int[]>();

            int maxCount = 0;

            for (int i = 0; i < mesh.Length; i++)
            {
                float meshTop = mesh[i].LocY + nodeRad;
                float meshBottom = mesh[i].LocY - nodeRad;
                float meshLeft = mesh[i].LocX - nodeRad;
                float meshRight = mesh[i].LocX + nodeRad;

                //List<int> bodies = new List<int>();

                List<int> bods = new List<int>();

                for (int b = 0; b < Bodies.Length; b++)
                {
                    if (Bodies[b].LocX < meshRight && Bodies[b].LocX > meshLeft && Bodies[b].LocY < meshTop && Bodies[b].LocY > meshBottom)
                    {
                        Bodies[b].MeshID = i;
                        mesh[i].Mass += Bodies[b].Mass;
                        mesh[i].Count++;
                        bods.Add(b);
                    }
                    else
                    {
                        bods.Add(-1);
                    }
                }

                if (mesh[i].Count > 0)
                {
                    if (mesh[i].Count > maxCount)
                        maxCount = mesh[i].Count;

                    meshList.Add(mesh[i]);
                    meshBods.Add(bods.ToArray());
                }


                //var bodyArr = meshBods.ToArray();
                //meshBodies[i] = bodyArr;
                // Array.Resize<int[,]>(ref meshBodies[i,], bodyArr.Length);
            }



            Mesh = meshList.ToArray();

            timer.Restart();

            MeshBodies = CreateRectangularArray(meshBods, maxCount);

            Console.WriteLine($@"Convert: {timer.ElapsedMilliseconds}");

            //Mesh = mesh;
            //MeshBodies = CreateRectangularArray(meshBods);

            //   Console.WriteLine(meshBodies.ToString());

        }

        static int[,] CreateRectangularArray(IList<int[]> arrays, int maxLen)
        {
            int minorLength = arrays[0].Length;

            int[,] ret = new int[arrays.Count, maxLen];

            for (int i = 0; i < arrays.Count; i++)
            {
                List<int> bods = new List<int>();
                int idx = 0;
                var array = arrays[i];
                if (array.Length != minorLength)
                {
                    throw new ArgumentException
                        ("All arrays must be the same length");
                }

                for (int j = 0; j < minorLength; j++)
                {
                    if (array[j] != -1)
                    {
                        bods.Add(array[j]);
                        //ret[i, idx] = array[j];
                        //idx++;
                    }
                }

                foreach (var bod in bods)
                {
                    ret[i, idx] = bod;
                    idx++;
                }

                for (int x = idx; x < maxLen; x++)
                {
                    ret[i, x] = -1;
                }

            }
            return ret;

            //int minorLength = arrays[0].Length;
            //T[,] ret = new T[arrays.Count, minorLength];
            //for (int i = 0; i < arrays.Count; i++)
            //{
            //    var array = arrays[i];
            //    if (array.Length != minorLength)
            //    {
            //        throw new ArgumentException
            //            ("All arrays must be the same length");
            //    }
            //    for (int j = 0; j < minorLength; j++)
            //    {
            //        ret[i, j] = array[j];
            //    }
            //}
            //return ret;
        }


        public static void RebuildUIDIndex()
        {
            UIDIndex.Clear();

            try
            {
                for (int i = 0; i < Bodies.Length; i++)
                {
                    UIDIndex.Add(Bodies[i].UID, i);
                }
            }
            catch
            {
                // Occational 'already added' race condition.
            }

        }

        public static PointF FollowBodyLoc()
        {
            if (UIDIndex.ContainsKey(FollowBodyUID))
                return new PointF((float)Bodies[UIDToIndex(FollowBodyUID)].LocX, (float)Bodies[UIDToIndex(FollowBodyUID)].LocY);
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

        public static PointF CenterOfMass()
        {
            double totMass = 0;

            for (int i = 0; i < Bodies.Length; i++)
            {
                var body = Bodies[i];

                totMass += body.Mass;
            }

            _totalMass = totMass;

            double cmX = 0, cmY = 0;

            for (int i = 0; i < Bodies.Length; i++)
            {
                var body = Bodies[i];

                cmX += body.Mass * body.LocX;
                cmY += body.Mass * body.LocY;
            }

            //cmX = (cmX / totMass) * -1f;
            //cmY = (cmY / totMass) * -1f;

            cmX = (cmX / totMass);
            cmY = (cmY / totMass);

            return new PointF((float)cmX, (float)cmY);
        }

        public static void TotEnergy()
        {
            double potE = 0;
            double kinE = 0;

            for (int i = 0; i < Bodies.Length; i++)
            {
                var bodyA = Bodies[i];

                kinE += bodyA.Mass * bodyA.AggregateSpeed() * bodyA.AggregateSpeed();

                for (int j = 0; j < Bodies.Length; j++)
                {
                    if (i != j)
                    {
                        var bodyB = Bodies[j];
                        double totMass = bodyA.Mass * bodyB.Mass;
                        double distX = bodyA.LocX - bodyB.LocX;
                        double distY = bodyA.LocY - bodyB.LocY;
                        double dist = (distX * distX) + (distY * distY);
                        double distSqrt = Math.Sqrt(dist);

                        potE += totMass / dist;
                        //potE += dist; //??

                        //kinE += 0.5f * (bo)
                    }
                }
            }

            kinE = 0.5f * kinE;
            potE = -0.5f * potE;

            Console.WriteLine($@"Kin: {kinE}  Pot: { potE}   tE: { (potE + kinE) }");
        }

        /// <summary>
        /// Calculate the orbital path of the specified body using only the current fields center of mass.  Very fast, but very inaccurate for bodies close to the center of mass.
        /// </summary>
        public static List<PointF> CalcPathCM(Body body)
        {
            var points = new List<PointF>();
            int segs = 500;
            float step = 0.300f;

            PointF speed = new PointF(body.SpeedX, body.SpeedY);
            PointF loc = new PointF(body.LocX, body.LocY);
            PointF force = new PointF();

            points.Add(loc);

            var bodiesCopy = new Body[Bodies.Length];
            Array.Copy(Bodies, bodiesCopy, bodiesCopy.Length);

            var cm = CenterOfMass();

            for (int i = 0; i < segs; i++)
            {
                force = new PointF();

                var distX = cm.X - loc.X;
                var distY = cm.Y - loc.Y;
                var dist = (distX * distX) + (distY * distY);
                var distSqrt = (float)Math.Sqrt(dist);

                var totMass = body.Mass * (_totalMass - body.Mass);
                var f = (float)totMass / (dist + 0.02f);

                force.X += (f * distX / distSqrt);
                force.Y += (f * distY / distSqrt);

                speed.X += step * force.X / body.Mass;
                speed.Y += step * force.Y / body.Mass;
                loc.X += step * (speed.X * 0.99f);
                loc.Y += step * (speed.Y * 0.99f);

                points.Add(loc);
            }

            return points;
        }

        /// <summary>
        /// Calculate the orbital path of the specified body by running a low res (large dt) static simulation against the current field. Accurate, but slow.
        /// </summary>
        public static List<PointF> CalcPath(Body body)
        {
            var points = new List<PointF>();
            int segs = 500;
            float dtStep = 0.100f;

            PointF speed = new PointF(body.SpeedX, body.SpeedY);
            PointF loc = new PointF(body.LocX, body.LocY);
            PointF force = new PointF();

            bool firstLoop = true;

            // Define a circle of influence around the specified body.
            // Bodies within this SOI are not included in orbit calculation.
            // This is done to improve accuracy by ignoring the neighbors of
            // a body within a large clump.
            var soi = new Ellipse(new PointF(body.LocX, body.LocY), 10);

            // This hashset will be used to cache SOI bodies for faster lookup on later loops.
            var soiBodies = new HashSet<int>();

            points.Add(loc);

            var bodiesCopy = new Body[Bodies.Length];
            Array.Copy(Bodies, bodiesCopy, bodiesCopy.Length);

            for (int i = 0; i < segs; i++)
            {
                force = new PointF();

                for (int b = 0; b < bodiesCopy.Length; b++)
                {
                    var bodyB = bodiesCopy[b];


                    if (bodyB.UID == body.UID)
                        continue;

                    if (body.HasCollision == 0)
                    {
                        var distX = bodyB.LocX - loc.X;
                        var distY = bodyB.LocY - loc.Y;
                        var dist = (distX * distX) + (distY * distY);
                        var distSqrt = (float)Math.Sqrt(dist);

                        var totMass = body.Mass * bodyB.Mass;

                        var f = totMass / (dist + 0.02f);

                        force.X += (f * distX / distSqrt);
                        force.Y += (f * distY / distSqrt);
                    }
                    else
                    {
                        // Use a slow "body is inside the circle" calculation on the first loop.
                        if (firstLoop)
                        {
                            // If this body is outside the SOI, calculate the forces.
                            if (!PointHelper.PointInsideCircle(soi.Location, soi.Size, (new PointF(bodyB.LocX, bodyB.LocY))))
                            {
                                var distX = bodyB.LocX - loc.X;
                                var distY = bodyB.LocY - loc.Y;
                                var dist = (distX * distX) + (distY * distY);
                                var distSqrt = (float)Math.Sqrt(dist);

                                var totMass = body.Mass * bodyB.Mass;

                                var f = totMass / (dist + 0.02f);

                                force.X += (f * distX / distSqrt);
                                force.Y += (f * distY / distSqrt);
                            }
                            else // If it is within the SOI, add to cache for faster lookup on the next loops.
                            {
                                soiBodies.Add(b);
                            }
                        }
                        else // After the first loop, use the hashset cache.
                        {
                            if (!soiBodies.Contains(b))
                            {
                                var distX = bodyB.LocX - loc.X;
                                var distY = bodyB.LocY - loc.Y;
                                var dist = (distX * distX) + (distY * distY);
                                var distSqrt = (float)Math.Sqrt(dist);

                                var totMass = body.Mass * bodyB.Mass;

                                var f = totMass / (dist + 0.02f);

                                force.X += (f * distX / distSqrt);
                                force.Y += (f * distY / distSqrt);
                            }
                        }
                    }
                }

                speed.X += dtStep * force.X / body.Mass;
                speed.Y += dtStep * force.Y / body.Mass;
                loc.X += dtStep * (speed.X * 0.99f);
                loc.Y += dtStep * (speed.Y * 0.99f);

                points.Add(loc);

                firstLoop = false;
            }

            return points;
        }

        /// <summary>
        /// Calculate the orbital path of the specified body by running a low res (large dt) static simulation against the current field. Accurate, but slow.
        /// 
        /// This variation tries to calculate a complete orbit instead of just advancing an N amount of steps.
        /// </summary>
        public static List<PointF> CalcPathCircle(Body body)
        {
            var points = new List<PointF>();
            int steps = 0;
            int maxSteps = 1000;//5000;
            float dtStep = 0.200f;
            bool complete = false;
            bool apoapsis = false;

            PointF speed = new PointF(body.SpeedX, body.SpeedY);
            PointF loc = new PointF(body.LocX, body.LocY);
            PointF force = new PointF();

            bool firstLoop = true;

            // Define a circle of influence around the specified body.
            // Bodies within this SOI are not included in orbit calculation.
            // This is done to improve accuracy by ignoring the neighbors of
            // a body within a large clump.
            var soi = new Ellipse(new PointF(body.LocX, body.LocY), 10);

            // This hashset will be used to cache SOI bodies for faster lookup on later loops.
            var soiBodies = new HashSet<int>();

            points.Add(loc);

            var bodiesCopy = new Body[Bodies.Length];
            Array.Copy(Bodies, bodiesCopy, bodiesCopy.Length);

            while (!complete)
            {
                force = new PointF();

                for (int b = 0; b < bodiesCopy.Length; b++)
                {
                    var bodyB = bodiesCopy[b];


                    if (bodyB.UID == body.UID)
                        continue;

                    if (body.HasCollision == 0)
                    {
                        var distX = bodyB.LocX - loc.X;
                        var distY = bodyB.LocY - loc.Y;
                        var dist = (distX * distX) + (distY * distY);
                        Console.WriteLine(dist);
                        var distSqrt = (float)Math.Sqrt(dist);

                        var totMass = body.Mass * bodyB.Mass;

                        var f = totMass / (dist + 0.02f);

                        force.X += (f * distX / distSqrt);
                        force.Y += (f * distY / distSqrt);
                    }
                    else
                    {
                        // Use a slow "body is inside the circle" calculation on the first loop.
                        if (firstLoop)
                        {
                            // If this body is outside the SOI, calculate the forces.
                            if (!PointHelper.PointInsideCircle(soi.Location, soi.Size, (new PointF(bodyB.LocX, bodyB.LocY))))
                            {
                                var distX = bodyB.LocX - loc.X;
                                var distY = bodyB.LocY - loc.Y;
                                var dist = (distX * distX) + (distY * distY);
                                var distSqrt = (float)Math.Sqrt(dist);

                                var totMass = body.Mass * bodyB.Mass;

                                var f = totMass / (dist + 0.02f);

                                force.X += (f * distX / distSqrt);
                                force.Y += (f * distY / distSqrt);
                            }
                            else // If it is within the SOI, add to cache for faster lookup on the next loops.
                            {
                                soiBodies.Add(b);
                            }
                        }
                        else // After the first loop, use the hashset cache.
                        {
                            if (!soiBodies.Contains(b))
                            {
                                var distX = bodyB.LocX - loc.X;
                                var distY = bodyB.LocY - loc.Y;
                                var dist = (distX * distX) + (distY * distY);
                                var distSqrt = (float)Math.Sqrt(dist);

                                var totMass = body.Mass * bodyB.Mass;

                                var f = totMass / (dist + 0.02f);

                                force.X += (f * distX / distSqrt);
                                force.Y += (f * distY / distSqrt);
                            }
                        }
                    }
                }

                speed.X += dtStep * force.X / body.Mass;
                speed.Y += dtStep * force.Y / body.Mass;
                loc.X += dtStep * (speed.X);
                loc.Y += dtStep * (speed.Y);

                points.Add(loc);

                firstLoop = false;

                if (!firstLoop && steps > 10)
                {
                    // Define a flat "plane" at the test body's Y coord.
                    var planeA = new PointF(-20000f, body.LocY);
                    var planeB = new PointF(20000f, body.LocY);

                    if (!apoapsis)
                    {
                        // Test for the first intersection of the plane.  This will be the apoapsis.
                        if (PointHelper.IsIntersecting(points[points.Count - 2], loc, planeA, planeB))
                        {
                            apoapsis = true;
                        }
                    }
                    else
                    {
                        // If we intersect the plane again after the apoapsis, we should now have a complete orbit.
                        if (PointHelper.IsIntersecting(points[points.Count - 2], loc, planeA, planeB))
                        {
                            complete = true;
                        }
                    }

                    // If we haven't found an expected orbit after the maximum steps, end the loop to display what was calculated.
                    if (steps >= maxSteps)
                    {
                        complete = true;
                    }

                }

                steps++;
            }

            return points;
        }


        public static bool IntersectsExisting(PointF location, float diameter)
        {
            float distX = 0;
            float distY = 0;
            float dist = 0;
            float colDist = 0;

            for (int i = 0; i < Bodies.Length; i++)
            {
                var body = Bodies[i];
                distX = body.LocX - location.X;
                distY = body.LocY - location.Y;
                dist = (distX * distX) + (distY * distY);
                colDist = (body.Size / 2f) + (diameter / 2f);

                if (dist <= (colDist * colDist))
                {
                    return true;
                }
            }

            return false;
        }

        public static void InsertExplosion(PointF location, int count)
        {
            MainLoop.WaitForPause();

            float lifetime = 0.04f;//0.1f;
            bool cloud = true;

            if (cloud)
            {
                var particles = new List<Body>();

                for (int i = 0; i < count; i++)
                {
                    var px = Numbers.GetRandomFloat(location.X - 0.5f, location.X + 0.5f);
                    var py = Numbers.GetRandomFloat(location.Y - 0.5f, location.Y + 0.5f);

                    while (!PointHelper.PointInsideCircle(location, 0.5f, new PointF(px, py)))
                    {
                        px = Numbers.GetRandomFloat(location.X - 0.5f, location.X + 0.5f);
                        py = Numbers.GetRandomFloat(location.Y - 0.5f, location.Y + 0.5f);
                    }

                    //particles.Add(NewBody(px, py, 1.5f, 1, Color.Orange, lifetime, 1));
                    particles.Add(NewBody(px, py, 1.0f, 1, Color.Orange, lifetime, 1));

                }

                Bodies = Bodies.Add(particles.ToArray());
            }
            else
            {
                Bodies = Bodies.Add(NewBody(location, 10, 400, Color.Orange, lifetime, 1));
            }

            MainLoop.Resume();
        }

        public static int NextUID()
        {
            _currentId++;
            return _currentId;
        }

        public static void Move(int index, PointF location)
        {
            if (index >= 0)
            {
                Bodies[index].LocX = location.X;
                Bodies[index].LocY = location.Y;
            }
        }

        public static void SetVelo(float velX, float velY)
        {
            for (int i = 0; i < Bodies.Length; i++)
            {
                Bodies[i].SpeedX = velX;
                Bodies[i].SpeedY = velY;
            }
        }

        public static void ShiftPos(float posX, float posY)
        {
            for (int i = 0; i < Bodies.Length; i++)
            {
                Bodies[i].LocX += posX;
                Bodies[i].LocY += posY;
            }
        }

        public static int Add(Body body)
        {
            body.UID = NextUID();
            Bodies = Bodies.Add(body);

            UIDIndex.Add(body.UID, Bodies.Length - 1);

            return body.UID;
        }

        public static T[] Add<T>(this T[] target, T item)
        {
            if (target == null)
            {
                //TODO: Return null or throw ArgumentNullException;
            }
            T[] result = new T[target.Length + 1];
            target.CopyTo(result, 0);
            result[target.Length] = item;
            return result;
        }

        public static T[] Add<T>(this T[] target, params T[] items)
        {
            // Validate the parameters
            if (target == null)
            {
                target = new T[] { };
            }
            if (items == null)
            {
                items = new T[] { };
            }

            // Join the arrays
            T[] result = new T[target.Length + items.Length];
            target.CopyTo(result, 0);
            items.CopyTo(result, target.Length);
            return result;
        }

        public static Body NewBody(float locX, float locY, float size, float mass, Color color, float lifetime, int isExplosion = 0)
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
            b.Lifetime = lifetime;
            b.Age = 0.0f;
            b.IsExplosion = isExplosion;

            if (isExplosion == 1)
            {
                b.InRoche = 1;
            }

            //b.DeltaTime = 0.0005f;

            b.BlackHole = 0;
            b.UID = NextUID();

            return b;
        }

        public static Body NewBody(PointF loc, float size, float mass, Color color, float lifetime, int isExplosion = 0)
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
            b.Lifetime = lifetime;
            b.Age = 0.0f;
            b.IsExplosion = isExplosion;
            b.BlackHole = 0;
            b.UID = NextUID();

            return b;
        }

        public static void Add(float locX, float locY, float size, float mass, Color color, float lifetime, int blackhole = 0)
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
            b.Lifetime = lifetime;
            b.Age = 0.0f;
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
            int index = Bodies.ToList().IndexOf(body);
            string info = $@"
Index: { index }
UID: { body.UID }
IsExplosion: { body.IsExplosion }
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
            // if (index > -1)
            Console.WriteLine(info);
        }
    }
}