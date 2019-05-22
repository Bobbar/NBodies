using NBodies.Extensions;
using NBodies.Helpers;
using NBodies.Rules;
using NBodies.Shapes;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using MessagePack;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public static class BodyManager
    {
        public static Body[] Bodies = new Body[0];
        //public static MeshCell[] Mesh = new MeshCell[0];
        public static MeshCell[] Mesh
        {
            get
            {
                return PhysicsProvider.PhysicsCalc.CurrentMesh;
            }
        }

        public static bool FollowSelected = false;
        public static int FollowBodyUID = -1;

        public static int BodyCount
        {
            get
            {
                return Bodies.Length;
            }
        }

        public static double TotalMass
        {
            get
            {
                return _totalMass;
            }
        }

        public static int StateIdx
        {
            get
            {
                if (_stateIdx < 0)
                    return 0;

                return _stateIdx;
            }
        }

        public static int StateCount
        {
            get
            {
                return _states.Count - 1;
            }
        }


        private static List<int> UIDBuckets = new List<int>(50000);
        private static int _currentUID = -1;
        private static double _totalMass = 0;
        private static int _stateIdx = -1;
        private const int _maxStates = 200;
        private static List<byte[]> _states = new List<byte[]>(_maxStates + 1);
        private static List<Body[]> _statesBuffer = new List<Body[]>(_maxStates + 1);
        private static bool _serializerRunning = false;
        private const float _timeSpan = 0.04f;
        private static float _elap = _timeSpan;

        public static void PushState(Body[] frame)
        {
            ResumeStates();

            // Don't save every frame to cut back on memory usage while also increasing length of time stored.
            if (_elap >= _timeSpan)
            {
                _statesBuffer.Add(frame);

                if (!_serializerRunning)
                    SerializeBufferAsync();

                _elap = 0f;
            }

            _elap += MainLoop.TimeStep;
        }

        private async static void SerializeBufferAsync()
        {
            await Task.Run(() =>
            {
                _serializerRunning = true;

                while (_statesBuffer.Count > 0)
                {
                    var state = _statesBuffer.First();

                    _states.Add(LZ4MessagePackSerializer.Serialize(state));

                    if (_states.Count - 1 > _maxStates)
                        _states.RemoveAt(0);

                    _stateIdx = _states.Count - 1;

                    if (_statesBuffer.Count > 0)
                        _statesBuffer.RemoveAt(0);
                }

                _serializerRunning = false;

            });
        }

        public static void PushState()
        {
            PushState(Bodies);
        }

        public static void RewindState()
        {
            if (_states.Count > 0)
            {
                if (_stateIdx - 1 >= 0)
                    _stateIdx--;

                Bodies = LZ4MessagePackSerializer.Deserialize<Body[]>(_states[_stateIdx]);

                RebuildUIDIndex();
            }

            _elap = _timeSpan;
        }

        public static void FastForwardState()
        {
            if (_states.Count > 0)
            {
                if (_stateIdx + 1 <= _states.Count - 1)
                    _stateIdx++;

                Bodies = LZ4MessagePackSerializer.Deserialize<Body[]>(_states[_stateIdx]);

                RebuildUIDIndex();
            }
        }

        public static void ResumeStates()
        {
            if (_stateIdx < _states.Count - 1 && _stateIdx != -1)
            {
                _states.RemoveRange(_stateIdx, _states.Count - _stateIdx);
                _stateIdx = -1;
            }
        }

        public static void ClearStates()
        {
            _states.Clear();
            _stateIdx = -1;
            _elap = _timeSpan;
        }

        /// <summary>
        /// Preallocated storage for culled bodies. Reduces one potentially large reallocation.
        /// </summary>
        private static Body[] _cullStore = new Body[500000];

        /// <summary>
        /// Culls invisible bodies, processes roche factures, rebuilds UID index.
        /// </summary>
        public static void PostProcess(bool processRoche)
        {
            if (Bodies.Length < 1) return;

            bool realloc = false;
            int position = 0;
            int newSize = 0;
            int maxUID = -1;
            double totMass = 0;
            List<Body> fractures = new List<Body>(0);

            // Make sure the preallocated store is large enough.
            if (_cullStore.Length < Bodies.Length)
                _cullStore = new Body[Bodies.Length];


            for (int i = 0; i < Bodies.Length; i++)
            {
                var body = Bodies[i];
                totMass += body.Mass;

                // Fracture large bodies in roche.
                if (processRoche)
                {
                    if (body.Size > 1)
                    {
                        if (body.Visible == 1 && body.InRoche == 1 && body.Flag != 1 && body.IsExplosion != 1)
                        {
                                body.Visible = 0;

                                if (fractures.Count == 0)
                                    fractures = new List<Body>(2000);

                                fractures.AddRange(FractureBody(Bodies[i]));
                        }
                    }
                }

                if (body.Visible == 0)
                {
                    // Only start to reallocate if we find an invisible body.
                    if (!realloc)
                    {
                        Array.Copy(Bodies, 0, _cullStore, 0, position);

                        realloc = true;
                        newSize = position;
                    }

                    // Stop following invisible bodies.
                    if (body.UID == FollowBodyUID)
                    {
                        FollowSelected = false;
                        FollowBodyUID = -1;
                    }

                }
                else
                {
                    // Store visible bodies in the a preallocated array.
                    if (realloc)
                    {
                        _cullStore[position] = body;
                        newSize++;
                    }

                    // Update the max UID.
                    maxUID = Math.Max(maxUID, body.UID);

                    // Update UID buckets and resize as needed.
                    if (body.UID < UIDBuckets.Count)
                    {
                        UIDBuckets[body.UID] = position;
                    }
                    else
                    {
                        int inc = (body.UID - UIDBuckets.Count) + 1;

                        UIDBuckets.AddRange(new int[inc]);
                        UIDBuckets[body.UID] = position;
                    }

                    // Update our current position for the cull store.
                    position++;
                }
            }

            // Set current UID to the determined max.
            _currentUID = maxUID;

            _totalMass = totMass;

            // Resize the main body array and copy from the cull store.
            if (realloc)
            {
                Array.Resize(ref Bodies, newSize);
                Array.Copy(_cullStore, 0, Bodies, 0, newSize);
            }

            // Add fractured bodies after to be processed on the following frame.
            if (fractures.Count > 0)
                Add(fractures.ToArray());

        }

        public static void CullDistant()
        {
            var cm = CenterOfMass();

            var bList = new List<Body>(Bodies);
            bList.RemoveAll(b =>
            {
                float distX = cm.X - b.PosX;
                float distY = cm.Y - b.PosY;
                float dist = distX * distX + distY * distY;

                if (dist > MainLoop.CullDistance * MainLoop.CullDistance)
                    return true;

                return false;

            });

            Bodies = bList.ToArray();
        }

        private static Body[] FractureBody(Body body)
        {
            float minSize = 1.0f;
            float newMass;
            float prevMass;

            bool flipflop = true;

            float density = body.Mass / (float)(Math.PI * Math.Pow(body.Size / 2, 2));

            newMass = CalcMass(1, density);

            int num = (int)(body.Mass / newMass);

            prevMass = body.Mass;

            var ellipse = new Ellipse(new PointF((float)body.PosX, (float)body.PosY), body.Size * 0.5f);

            bool done = false;
            float stepSize = minSize * 0.98f;

            float startXpos = ellipse.Location.X - ellipse.Size;
            float startYpos = ellipse.Location.Y - ellipse.Size;

            float Xpos = startXpos;
            float Ypos = startYpos;

            var newBodies = new List<Body>();

            int its = 0;

            while (!done)
            {
                var testPoint = new PointF(Xpos, Ypos);

                if (PointExtensions.PointInsideCircle(ellipse.Location, ellipse.Size, testPoint))
                {
                    var newbody = NewBody(testPoint.X, testPoint.Y, body.VeloX, body.VeloY, minSize, newMass, Color.FromArgb(body.Color), 1);
                    newbody.ForceTot = body.ForceTot;
                    newBodies.Add(newbody);
                }

                Xpos += stepSize;

                if (Xpos > ellipse.Location.X + (ellipse.Size))
                {
                    if (flipflop)
                    {
                        Xpos = startXpos + (minSize / 2f);
                        flipflop = false;
                    }
                    else
                    {
                        Xpos = startXpos;
                        flipflop = true;
                    }

                    Ypos += stepSize - 0.20f;
                }

                if (newBodies.Count == num || its > num * 4)
                {
                    done = true;
                }

                its++;
            }

            return newBodies.ToArray();
        }

        public static double UpdateTotMass()
        {
            double tMass = 0;

            for (int i = 0; i < Mesh.Length; i++)
            {
                tMass += Mesh[i].Mass;
            }

            _totalMass = tMass;

            return _totalMass;
        }

        public static void ClearBodies()
        {
            FollowSelected = false;
            FollowBodyUID = -1;
            Bodies = new Body[0];

            UIDBuckets.Clear();

            _currentUID = -1;
            ClearStates();

            MainLoop.FrameCount = 0;
            MainLoop.TotalTime = 0;
        }

        public static void ReplaceBodies(Body[] bodies)
        {
            ClearBodies();

            _currentUID = bodies.Max((b) => b.UID);

            Bodies = bodies;

            CullDistant();

            RebuildUIDIndex();
        }

        public static void RebuildUIDIndex()
        {
            var maxUID = Bodies.Max(b => b.UID);

            UIDBuckets = new List<int>(new int[maxUID + 1]);
            _currentUID = maxUID;

            for (int i = 0; i < Bodies.Length; i++)
            {
                UIDBuckets[Bodies[i].UID] = i;
            }
        }

        //public static PointF FollowBodyLoc()
        //{
        //    if (UIDIndex.ContainsKey(FollowBodyUID))
        //        return new PointF((float)Bodies[UIDToIndex(FollowBodyUID)].PosX, (float)Bodies[UIDToIndex(FollowBodyUID)].PosY);
        //    return new PointF();
        //}

        public static Body FollowBody()
        {
            try
            {
                if (FollowSelected)
                {
                    return Bodies[UIDToIndex(FollowBodyUID)];
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
            if (uid < UIDBuckets.Count)
                return UIDBuckets[uid];

            return 0;
        }

        public static Body BodyFromUID(int uid)
        {
            int idx = UIDToIndex(uid);

            if (idx < Bodies.Length)
                return Bodies[idx];

            return Bodies[0];
        }

        public static void AddUID(int uid, int idx)
        {
            if (uid < UIDBuckets.Count)
            {
                UIDBuckets[uid] = idx;
            }
            else if (uid == UIDBuckets.Count)
            {
                UIDBuckets.Add(idx);
            }
            else
            {
                RebuildUIDIndex();
            }
        }

        public static PointF MeshCenterOfMass()
        {
            double totMass = 0;
            double cmX = 0, cmY = 0;

            for (int i = PhysicsProvider.PhysicsCalc.LevelIndex[MainLoop.MeshLevels]; i < Mesh.Length; i++)
            {
                var mesh = Mesh[i];

                totMass += mesh.Mass;
                cmX += mesh.Mass * mesh.CmX;
                cmY += mesh.Mass * mesh.CmY;
            }

            cmX = (cmX / totMass);
            cmY = (cmY / totMass);

            return new PointF((float)cmX, (float)cmY);
        }

        public static PointF CenterOfMass()
        {
            double totMass = 0;
            double cmX = 0, cmY = 0;

            for (int i = 0; i < Bodies.Length; i++)
            {
                var body = Bodies[i];

                totMass += body.Mass;

                cmX += body.Mass * body.PosX;
                cmY += body.Mass * body.PosY;
            }

            cmX = (cmX / totMass);
            cmY = (cmY / totMass);

            return new PointF((float)cmX, (float)cmY);
        }

        public static void TotEnergy()
        {
            float E = 0, d = 0;

            for (int i = 0; i < Bodies.Length; i++)
            {
                var body = Bodies[i];

                E += 0.5f * body.Mass * body.Velocity().Length();

                for (int j = 0; j < Bodies.Length; j++)
                {
                    if (i != j)
                    {
                        var bodyB = Bodies[j];
                        d = body.Position().DistanceSqrt(bodyB.Position());

                        E -= 1.488E-34f * body.Mass * bodyB.Mass / d;
                    }
                }
            }

            Console.WriteLine($@"Energy: {E}");
        }

        /// <summary>
        /// Calculate the orbital path of the specified body using only the current fields center of mass.  Very fast, but very inaccurate for bodies close to the center of mass.
        /// </summary>
        public static List<PointF> CalcPathCM(Body body)
        {
            var points = new List<PointF>();
            int segs = 500;
            float step = 0.300f;

            PointF speed = new PointF(body.VeloX, body.VeloY);
            PointF loc = new PointF(body.PosX, body.PosY);
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
        ///
        /// This variation tries to calculate a complete orbit instead of just advancing an N amount of steps.
        /// </summary>
        ///
        public static List<PointF> CalcPathCircle(Body body)
        {
            if (Mesh.Length < 1)
                return new List<PointF>();

            var points = new List<PointF>();
            int steps = 0;
            int maxSteps = 1000;//5000;
            float dtStep = 0.200f;
            bool complete = false;
            bool apoapsis = false;
            int start = 0;
            int end = 0;
            int meshLevel = 3;
            int[] levelIndex = PhysicsProvider.PhysicsCalc.LevelIndex;

            if (meshLevel >= levelIndex.Length)
                meshLevel = levelIndex.Length - 1;

            start = levelIndex[meshLevel];
            end = (meshLevel + 1 >= levelIndex.Length) ? Mesh.Length : levelIndex[meshLevel + 1];

            PointF speed = new PointF(body.VeloX, body.VeloY);
            PointF loc = new PointF(body.PosX, body.PosY);
            PointF startLoc = loc;
            PointF planeA = new PointF();
            PointF planeB = new PointF();

            points.Add(loc);

            try
            {
                while (!complete)
                {
                    float forceX = 0;
                    float forceY = 0;

                    for (int c = start; c < end; c++)
                    {
                        if (end > Mesh.Length)
                            break;

                        var cell = Mesh[c];

                        if (cell.ID == body.MeshID)
                            continue;

                        var distX = cell.CmX - loc.X;
                        var distY = cell.CmY - loc.Y;
                        var dist = (distX * distX) + (distY * distY);
                        var distSqrt = (float)Math.Sqrt(dist);

                        if (distSqrt > (cell.Size))
                        {
                            var totMass = body.Mass * cell.Mass;

                            var f = totMass / (dist + 0.02f);

                            forceX += (float)(f * distX / distSqrt);
                            forceY += (float)(f * distY / distSqrt);
                        }

                    }

                    speed.X += dtStep * forceX / body.Mass;
                    speed.Y += dtStep * forceY / body.Mass;
                    loc.X += dtStep * (speed.X);
                    loc.Y += dtStep * (speed.Y);

                    points.Add(loc);

                    if (steps == 0)
                    {
                        // Define a flat "plane" perpendicular to the first 2 points.
                        var len = 20000;
                        planeA = TangentPoint(startLoc, points[1], -len);
                        planeB = TangentPoint(startLoc, points[1], len);
                    }

                    if (steps > 10)
                    {
                        if (!apoapsis)
                        {
                            // Test for the first intersection of the plane.  This will be the apoapsis.
                            if (PointExtensions.IsIntersecting(points[points.Count - 2], loc, planeA, planeB))
                            {
                                apoapsis = true;
                            }
                        }
                        else
                        {
                            // If we intersect the plane again after the apoapsis, we should now have a complete orbit.
                            if (PointExtensions.IsIntersecting(points[points.Count - 2], loc, planeA, planeB))
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
            }
            catch
            {
                return new List<PointF>();
            }


            return points;
        }

        private static PointF TangentPoint(PointF a, PointF b, float len)
        {
            var l1 = a;
            var l2 = b;

            double slope = (l2.Y - l1.Y) / (l2.X - l1.X);
            slope = -1 / slope;

            float midX = (l1.X + l2.X) / 2;
            float midY = (l1.Y + l2.Y) / 2;

            double c = -slope * midX + midY;

            return new PointF(midX + len, (float)(slope * (midX + len) + c));
        }

        //public static ColoredLine[] GetInteractions(Body body)
        //{
        //    float distX, distY, dist;

        //    var lineList = new List<ColoredLine>();

        //    MeshCell bodyCell = Mesh[body.MeshID];

        //    var bodyLoc = new PointF(body.LocX, body.LocY);

        //    for (int c = 0; c < Mesh.Length; c++)
        //    {
        //        MeshCell cell = Mesh[c];
        //        var cellLoc = new PointF(cell.CmX, cell.CmY);

        //        //distX = cell.LocX - body.LocX;
        //        //distY = cell.LocY - body.LocY;
        //        distX = cell.CmX - body.LocX;
        //        distY = cell.CmY - body.LocY;
        //        dist = (distX * distX) + (distY * distY);

        //        float maxDist = cell.PPDist + bodyCell.PPDist;//ppDist;

        //        if (dist > maxDist * maxDist && cell.ID != body.MeshID)
        //        {
        //            //mesh line;
        //            lineList.Add(new ColoredLine(Color.LawnGreen, bodyLoc, cellLoc));
        //        }
        //        else
        //        {
        //            int mbLen = cell.BodCount;
        //            for (int mb = 0; mb < mbLen; mb++)
        //            {
        //                int meshBodId = MeshBodies[c, mb];

        //                Body cellBod = Bodies[meshBodId];
        //                var cellBodLoc = new PointF(cellBod.LocX, cellBod.LocY);

        //                if (cellBod.UID != body.UID)
        //                {
        //                    // body line
        //                    lineList.Add(new ColoredLine(Color.Red, bodyLoc, cellBodLoc));

        //                }
        //            }
        //        }
        //    }

        //    return lineList.ToArray();
        //}

        public static void InsertExplosion(PointF location, int count)
        {
            MainLoop.WaitForPause();

            float lifetime = 0.08f;//0.1f;
            bool cloud = true;

            if (cloud)
            {
                var particles = new List<Body>();

                for (int i = 0; i < count; i++)
                {
                    var px = Numbers.GetRandomFloat(location.X - 0.5f, location.X + 0.5f);
                    var py = Numbers.GetRandomFloat(location.Y - 0.5f, location.Y + 0.5f);

                    while (!PointExtensions.PointInsideCircle(location, 0.5f, new PointF(px, py)))
                    {
                        px = Numbers.GetRandomFloat(location.X - 0.5f, location.X + 0.5f);
                        py = Numbers.GetRandomFloat(location.Y - 0.5f, location.Y + 0.5f);
                    }

                    particles.Add(NewBody(px, py, 1.0f, 1, Color.Orange, lifetime, 1));
                }

                Bodies = Bodies.Add(particles.ToArray());
            }
            else
            {
                Bodies = Bodies.Add(NewBody(location, 10, 400, Color.Orange, lifetime, 1));
            }

            MainLoop.ResumePhysics(true);
        }

        public static int NextUID()
        {
            _currentUID++;
            return _currentUID;
        }

        public static void Move(int index, PointF location)
        {
            if (index >= 0)
            {
                Bodies[index].PosX = location.X;
                Bodies[index].PosY = location.Y;
            }
        }

        public static void SetVelo(float velX, float velY)
        {
            for (int i = 0; i < Bodies.Length; i++)
            {
                Bodies[i].VeloX = velX;
                Bodies[i].VeloY = velY;
            }
        }

        public static void ShiftPos(float posX, float posY)
        {
            for (int i = 0; i < Bodies.Length; i++)
            {
                Bodies[i].PosX += posX;
                Bodies[i].PosY += posY;
            }
        }

        public static int Add(Body body)
        {
            body.UID = NextUID();
            Bodies = Bodies.Add(body);

            AddUID(body.UID, Bodies.Length - 1);

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

        public static Body NewBody(float locX, float locY, float veloX, float veloY, float size, float mass, Color color, int inRoche)
        {
            var b = new Body();

            b.PosX = locX;
            b.PosY = locY;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.VeloX = veloX;
            b.VeloY = veloY;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = inRoche;
            b.Lifetime = 0;
            b.Age = 0.0f;
            b.IsExplosion = 0;

            b.Flag = 0;
            b.UID = NextUID();

            return b;
        }

        public static Body NewBody(float locX, float locY, float veloX, float veloY, float size, float mass, Color color)
        {
            var b = new Body();

            b.PosX = locX;
            b.PosY = locY;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.VeloX = veloX;
            b.VeloY = veloY;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = 0;
            b.Lifetime = 0;
            b.Age = 0.0f;
            b.IsExplosion = 0;

            b.Flag = 0;
            b.UID = NextUID();

            return b;
        }

        public static Body NewBody(float locX, float locY, float size, float mass, Color color, float lifetime, int isExplosion = 0)
        {
            var b = new Body();

            b.PosX = locX;
            b.PosY = locY;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.VeloX = 0;
            b.VeloY = 0;
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

            b.Flag = 0;
            b.UID = NextUID();

            return b;
        }

        public static Body NewBody(PointF loc, float size, float mass, Color color, float lifetime, int isExplosion = 0)
        {
            var b = new Body();

            b.PosX = loc.X;
            b.PosY = loc.Y;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.VeloX = 0;
            b.VeloY = 0;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = 0;
            b.Lifetime = lifetime;
            b.Age = 0.0f;
            b.IsExplosion = isExplosion;
            b.Flag = 0;
            b.UID = NextUID();

            return b;
        }

        public static Body NewBody(PointF loc, float size, float mass, Color color, int isBlackhole = 0)
        {
            var b = new Body();

            b.PosX = loc.X;
            b.PosY = loc.Y;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.VeloX = 0;
            b.VeloY = 0;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = 0;
            b.Lifetime = 0;
            b.Age = 0.0f;
            b.IsExplosion = 0;
            b.Flag = isBlackhole;
            b.UID = NextUID();

            return b;
        }

        public static void Add(float locX, float locY, float size, float mass, Color color, float lifetime, int blackhole = 0)
        {
            var b = new Body();

            b.PosX = locX;
            b.PosY = locY;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.VeloX = 0;
            b.VeloY = 0;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = 0;
            b.Lifetime = lifetime;
            b.Age = 0.0f;
            b.Flag = blackhole;
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

        public static void Add(Body[] bodies)
        {
            Bodies = Bodies.Add(bodies);
        }

        public static void Add(float locX, float locY, float velX, float velY, float size, float mass, Color color)
        {
            var b = new Body();

            b.PosX = locX;
            b.PosY = locY;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.VeloX = velX;
            b.VeloY = velY;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = 0;
            b.Flag = 0;
            b.UID = -1;

            Add(b);
        }

        public static void Add(float locX, float locY, float velX, float velY, float size, float mass, Color color, int inRoche)
        {
            var b = new Body();

            b.PosX = locX;
            b.PosY = locY;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.VeloX = velX;
            b.VeloY = velY;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = inRoche;
            b.Flag = 0;
            b.UID = -1;

            Add(b);
        }

        public static void Add(PointF loc, float size, float mass, Color color, int blackhole = 0)
        {
            var b = new Body();

            b.PosX = loc.X;
            b.PosY = loc.Y;
            b.Mass = mass;
            b.Size = size;
            b.Color = color.ToArgb();

            b.VeloX = 0;
            b.VeloY = 0;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = 0;
            b.Flag = blackhole;
            b.UID = -1;

            Add(b);
        }

        public static int Add(PointF loc, float size, Color color)
        {
            var b = new Body();

            b.PosX = loc.X;
            b.PosY = loc.Y;
            b.Mass = CalcMass(size);
            b.Size = size;
            b.Color = color.ToArgb();

            b.VeloX = 0;
            b.VeloY = 0;
            b.ForceX = 0;
            b.ForceY = 0;
            b.ForceTot = 0;
            b.Visible = 1;
            b.InRoche = 0;
            b.Flag = 0;
            b.UID = -1;
            b.MeshID = -1;

            return Add(b);
        }

        public static float CalcMass(float size)
        {
            return (float)(Math.PI * (float)(Math.Pow(size / 2.0f, 2))) * Matter.Density;
        }

        public static float CalcMass(float size, float density)
        {
            return (float)(Math.PI * (Math.Pow(size / 2.0f, 2))) * density;
        }

        public static float CalcRadius(float area)
        {
            return (float)Math.Sqrt(area / Math.PI);
        }

        public static float AggregateSpeed(this Body body)
        {
            return (float)Math.Sqrt(Math.Pow(body.VeloX, 2) + Math.Pow(body.VeloY, 2));
        }

        public static PointF Velocity(this Body body)
        {
            return new PointF(body.VeloX, body.VeloY);
        }

        public static PointF Position(this Body body)
        {
            return new PointF((float)body.PosX, (float)body.PosY);
        }

        public static void PrintInfo(this Body body)
        {
            MeshCell mesh = new MeshCell();
            if (body.MeshID != -1 && (body.MeshID <= Mesh.Length - 1))
            {
                mesh = Mesh[body.MeshID];
            }
            int index = Bodies.ToList().IndexOf(body);
            string info = $@"
Index: { index }
UID: { body.UID }
MeshID: { body.MeshID }
    Count: { mesh.BodyCount }
    Mass: { mesh.Mass }
    Neighbors: { mesh.NeighborCount }
    Cm (X,Y): { mesh.CmX }, { mesh.CmY }
    Loc (X,Y): { mesh.LocX }, { mesh.LocY }
    Idx (X,Y): { mesh.IdxX }, { mesh.IdxY }

IsExplosion: { body.IsExplosion }
Mass: { body.Mass }
Size: { body.Size }
InRoche: { body.InRoche }
Density: { body.Density }
Pressure: { body.Pressure }
Agg. Speed: { body.AggregateSpeed() }
Speed (X,Y): { body.VeloX }, { body.VeloY }
Position (X,Y): { body.PosX }, { body.PosY }
Force (X,Y): { body.ForceX }, { body.ForceY }
Tot Force: { body.ForceTot }
";
            // if (index > -1)
            Console.WriteLine(info);
        }
    }
}