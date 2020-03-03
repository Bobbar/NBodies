using NBodies.Extensions;
using NBodies.Rules;
using NBodies.Shapes;
using NBodies.Physics;
using NBodies.Helpers;
using System;
using System.Drawing;
using System.Windows.Forms;
using System.Collections.Generic;
using OpenTK;

namespace NBodies
{
    public partial class AddBodiesForm : Form
    {
        private Color _pickedColor = Color.Transparent;
        private bool _colorWasPicked = false;
        private Vector3 _centerLocation;

        public AddBodiesForm()
        {
            InitializeComponent();
            DensityTextBox.Text = Rules.Matter.Density.ToString();
        }

        private float CircleV(float rx, float ry, float centerMass)
        {
            float r2 = (float)Math.Sqrt(rx * rx + ry * ry);
            float numerator = (float)(0.977 * centerMass);
            return (float)Math.Sqrt(numerator / r2);
        }

        private PointF OrbitVel(PointF bodyALoc, PointF bodyBLoc, float bodyAMass)
        {
            var offsetP = bodyBLoc;
            offsetP = offsetP.Subtract(bodyALoc);

            float magV = CircleV(offsetP.X, offsetP.Y, bodyAMass);
            float absAngle = (float)Math.Atan(Math.Abs(offsetP.Y / offsetP.X));
            float thetaV = (float)Math.PI * 0.5f - absAngle;
            float vx = -1 * (float)(Math.Sign(offsetP.Y) * Math.Cos(thetaV) * magV);
            float vy = (float)(Math.Sign(offsetP.X) * Math.Sin(thetaV) * magV);

            return new PointF(vx, vy);
        }

        private Vector3 OrbitVel(Vector3 bodyALoc, Vector3 bodyBLoc, float bodyAMass)
        {
            var offsetP = bodyBLoc - bodyALoc;

            float magV = CircleV(offsetP.X, offsetP.Y, bodyAMass);
            float absAngle = (float)Math.Atan(Math.Abs(offsetP.Y / offsetP.X));
            float thetaV = (float)Math.PI * 0.5f - absAngle;
            float vx = -1 * (float)(Math.Sign(offsetP.Y) * Math.Cos(thetaV) * magV);
            float vy = (float)(Math.Sign(offsetP.X) * Math.Sin(thetaV) * magV);

            return new Vector3(vx, vy, 0);
        }

        private void AddBodiesToOrbit(int count, int maxSize, int minSize, int bodyMass, bool includeCenterMass, float centerMass)
        {
            MainLoop.WaitForPause();

            _centerLocation = ViewportHelpers.CameraPos;

            var newBodies = new List<Body>();
            float px, py, pz;
            float radius = float.Parse(OrbitRadiusTextBox.Text);
            float innerRadius = float.Parse(InOrbitRadiusTextBox.Text.Trim());

            Rules.Matter.Density = float.Parse(DensityTextBox.Text);
            centerMass *= Rules.Matter.Density * 2;

            var ellipse = new Ellipse3D(_centerLocation, radius);
            var inEllipse = new Ellipse3D(_centerLocation, innerRadius);

            var sysVelo = new Vector3(Convert.ToSingle(VeloXText.Text.Trim()), Convert.ToSingle(VeloYText.Text.Trim()), Convert.ToSingle(VeloZText.Text.Trim()));

            if (includeCenterMass)
            {
                var cm = BodyManager.NewBody(ellipse.Location, 3, centerMass, Color.Black, 1);
                cm.VeloX += sysVelo.X;
                cm.VeloY += sysVelo.Y;
                cm.VeloZ += sysVelo.Z;

                newBodies.Add(cm);
            }

            for (int i = 0; i < count; i++)
            {
                var bodySize = Numbers.GetRandomFloat(minSize, maxSize);
                px = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                py = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);
                pz = Numbers.GetRandomFloat(ellipse.Location.Z - 10, ellipse.Location.Z + 10);

                int its = 0;
                int maxIts = 100;
                bool outOfSpace = false;

                var newLoc = new Vector3(px, py, pz);

                while (!PointExtensions.PointInsideCircle(ellipse.Location, ellipse.Size, newLoc) || PointExtensions.PointInsideCircle(inEllipse.Location, inEllipse.Size, newLoc))
                {
                    if (its >= maxIts)
                    {
                        Console.WriteLine("Failed to add body after allotted tries!");
                        outOfSpace = true;
                        break;
                    }

                    bodySize = Numbers.GetRandomFloat(minSize, maxSize);
                    px = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                    py = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);
                    pz = Numbers.GetRandomFloat(ellipse.Location.Z - 10, ellipse.Location.Z + 10);

                    newLoc = new Vector3(px, py, pz);
                    its++;
                }

                if (outOfSpace)
                    continue;

                var bodyVelo = OrbitVel(ellipse.Location, newLoc, centerMass);

                float newMass = 1;

                bool byDist = layeredCheckBox.Checked;
                MatterType matter;

                if (byDist)
                {
                    var dist = Vector3.Distance(newLoc, ellipse.Location);
                    matter = Matter.GetForDistance(dist, radius);
                }
                else
                {
                    matter = Matter.GetRandom();
                }

                if (StaticDensityCheckBox.Checked)
                {
                    if (bodyMass > 0)
                    {
                        newMass = bodyMass;
                    }
                    else
                    {
                        newMass = BodyManager.CalcMass(bodySize);
                    }
                }
                else
                {
                    newMass = BodyManager.CalcMass(bodySize, matter.Density);
                }

                Color color;
                if (StaticDensityCheckBox.Checked)
                {
                    if (_colorWasPicked)
                        color = _pickedColor;
                    else
                        color = ColorHelper.RandomColor();
                }
                else
                {
                    color = matter.Color;
                }

                var velo = new Vector3(bodyVelo.X, bodyVelo.Y, 0);
                velo += sysVelo;
                newBodies.Add(BodyManager.NewBody(newLoc, velo, bodySize, newMass, color));
            }

            var bodyArr = newBodies.ToArray();

            if (fixOverlapCheckBox.Checked)
                FixOverlaps(ref bodyArr, 3);

            float rotX = Convert.ToSingle(RotXText.Text.Trim());
            float rotY = Convert.ToSingle(RotYText.Text.Trim());
            float rotZ = Convert.ToSingle(RotZText.Text.Trim());

            var rot = Matrix4.Identity;
            rot *= Matrix4.CreateRotationX(rotX);
            rot *= Matrix4.CreateRotationY(rotY);
            rot *= Matrix4.CreateRotationZ(rotZ);

            for (int i = 0; i < bodyArr.Length; i++)
            {
                var body = bodyArr[i];
                var pos = new Vector4(body.PositionVec(), 1);
                var velo = new Vector4(body.VelocityVec(), 1);
                pos *= rot;
                velo *= rot;

                body.PosX = pos.X;
                body.PosY = pos.Y;
                body.PosZ = pos.Z;

                body.VeloX = velo.X;
                body.VeloY = velo.Y;
                body.VeloZ = velo.Z;

                bodyArr[i] = body;
            }

            BodyManager.Add(bodyArr);

            MainLoop.ResumePhysics();
        }

        private void AddBodiesToDisc(int count, int maxSize, int minSize, int bodyMass)
        {
            MainLoop.WaitForPause();

            var newBodies = new List<Body>();

            float px, py, pz;
            float radius = float.Parse(OrbitRadiusTextBox.Text.Trim());
            float innerRadius = float.Parse(InOrbitRadiusTextBox.Text.Trim());

            Matter.Density = float.Parse(DensityTextBox.Text);
            var ellipse = new Ellipse3D(ViewportHelpers.CameraPos, radius);
            var inEllipse = new Ellipse3D(ViewportHelpers.CameraPos, innerRadius);

            for (int i = 0; i < count; i++)
            {
                var bodySize = Numbers.GetRandomFloat(minSize, maxSize);
                px = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                py = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);
                pz = Numbers.GetRandomFloat(ellipse.Location.Z - ellipse.Size, ellipse.Location.Z + ellipse.Size);

                int its = 0;
                int maxIts = 100;
                bool outOfSpace = false;

                Vector3 newLoc = new Vector3(px, py, pz);

                while (!PointExtensions.PointInsideCircle(ellipse.Location, ellipse.Size, newLoc) || PointExtensions.PointInsideCircle(inEllipse.Location, inEllipse.Size, newLoc))
                {
                    if (its >= maxIts)
                    {
                        Console.WriteLine("Failed to add body after allotted tries!");
                        outOfSpace = true;
                        break;
                    }

                    bodySize = Numbers.GetRandomFloat(minSize, maxSize);
                    px = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                    py = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);
                    pz = Numbers.GetRandomFloat(ellipse.Location.Z - ellipse.Size, ellipse.Location.Z + ellipse.Size);

                    newLoc = new Vector3(px, py, pz);

                    its++;
                }

                if (outOfSpace)
                    continue;

                float newMass;
                bool byDist = layeredCheckBox.Checked;
                MatterType matter;

                if (byDist)
                {
                    var dist = Vector3.Distance(newLoc, ellipse.Location);
                    matter = Matter.GetForDistance(dist, radius);
                }
                else
                {
                    matter = Matter.GetRandom();
                }


                if (StaticDensityCheckBox.Checked)
                {
                    if (bodyMass > 0)
                    {
                        newMass = bodyMass;
                    }
                    else
                    {
                        newMass = BodyManager.CalcMass(bodySize);
                    }
                }
                else
                {
                    newMass = BodyManager.CalcMass(bodySize, matter.Density);
                }

                Color color;
                if (StaticDensityCheckBox.Checked)
                {
                    if (_colorWasPicked)
                        color = _pickedColor;
                    else
                        color = ColorHelper.RandomColor();
                }
                else
                {
                    color = matter.Color;
                }

                //  float rndZ = Numbers.GetRandomFloat(-100, 100);
                newBodies.Add(BodyManager.NewBody(newLoc.X, newLoc.Y, newLoc.Z, bodySize, newMass, color, int.Parse(LifeTimeTextBox.Text.Trim())));

                //newBodies.Add(BodyManager.NewBody(newLoc.X, newLoc.Y, bodySize, newMass, color, int.Parse(LifeTimeTextBox.Text.Trim())));
            }

            var bodyArr = newBodies.ToArray();

            if (fixOverlapCheckBox.Checked)
                FixOverlaps(ref bodyArr, 3);

            BodyManager.Add(bodyArr);

            MainLoop.ResumePhysics();
        }

        private bool IntersectsExisting(List<Body> bodies, PointF location, float diameter)
        {
            float distX = 0;
            float distY = 0;
            float dist = 0;
            float colDist = 0;

            for (int i = 0; i < bodies.Count; i++)
            {
                var body = bodies[i];
                distX = body.PosX - location.X;
                distY = body.PosY - location.Y;
                dist = (distX * distX) + (distY * distY);
                colDist = (body.Size / 2f) + (diameter / 2f);

                if (dist <= (colDist * colDist))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// OpenCL accelerated overlap correcting.
        /// </summary>
        private void FixOverlaps(ref Body[] bodies, int passes)
        {
            for (int i = 0; i < passes; i++)
            {
                PhysicsProvider.PhysicsCalc.FixOverLaps(ref bodies);
            }
        }

        private void AddOrbitButton_Click(object sender, EventArgs e)
        {
            AddBodiesToOrbit(int.Parse(NumToAddTextBox.Text.Trim()), int.Parse(MaxSizeTextBox.Text.Trim()), int.Parse(MinSizeTextBox.Text.Trim()), int.Parse(MassTextBox.Text.Trim()), CenterMassCheckBox.Checked, float.Parse(CenterMassTextBox.Text.Trim()));
        }
        private void AddStationaryButton_Click(object sender, EventArgs e)
        {
            AddBodiesToDisc(int.Parse(NumToAddTextBox.Text.Trim()), int.Parse(MaxSizeTextBox.Text.Trim()), int.Parse(MinSizeTextBox.Text.Trim()), int.Parse(MassTextBox.Text.Trim()));
        }

        private void StaticDensityCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            layeredCheckBox.Enabled = !StaticDensityCheckBox.Checked;
            DensityTextBox.Enabled = !StaticDensityCheckBox.Checked;
            PickColorButton.Enabled = StaticDensityCheckBox.Checked;
        }

        private void PickColorButton_Click(object sender, EventArgs e)
        {
            using (var colorPick = new ColorDialog())
            {
                if (colorPick.ShowDialog(this) == DialogResult.OK)
                {
                    _pickedColor = colorPick.Color;
                    _colorWasPicked = true;
                }
                else
                {
                    _pickedColor = Color.Transparent;
                    _colorWasPicked = false;
                }
            }
        }
    }
}
