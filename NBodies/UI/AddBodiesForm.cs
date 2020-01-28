using NBodies.Extensions;
using NBodies.Rules;
using NBodies.Shapes;
using NBodies.Physics;
using NBodies.Helpers;
using System;
using System.Drawing;
using System.Windows.Forms;
using System.Collections.Generic;

namespace NBodies
{
    public partial class AddBodiesForm : Form
    {
        private Color _pickedColor = Color.Transparent;
        private bool _colorWasPicked = false;

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

        private void AddBodiesToOrbit(int count, int maxSize, int minSize, int bodyMass, bool includeCenterMass, float centerMass)
        {
            MainLoop.WaitForPause();

            var newBodies = new List<Body>();
            float px, py;
            float radius = float.Parse(OrbitRadiusTextBox.Text);
            float innerRadius = float.Parse(InOrbitRadiusTextBox.Text.Trim());

            Rules.Matter.Density = float.Parse(DensityTextBox.Text);
            centerMass *= Rules.Matter.Density * 2;

            var ellipse = new Ellipse(ViewportHelpers.ScreenPointToField(ViewportOffsets.ScreenCenter), radius);
            var inEllipse = new Ellipse(ViewportHelpers.ScreenPointToField(ViewportOffsets.ScreenCenter), innerRadius);

            if (includeCenterMass)
            {
                newBodies.Add(BodyManager.NewBody(ellipse.Location, 3, centerMass, Color.Black, 1));
            }

            for (int i = 0; i < count; i++)
            {
                var bodySize = Numbers.GetRandomFloat(minSize, maxSize);
                px = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                py = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);

                int its = 0;
                int maxIts = 100;
                bool outOfSpace = false;

                PointF newLoc = new PointF(px, py);

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
                    newLoc = new PointF(px, py);
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
                    var dist = newLoc.DistanceSqrt(ellipse.Location);
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

                newBodies.Add(BodyManager.NewBody(px, py, bodyVelo.X, bodyVelo.Y, bodySize, newMass, color));
            }

            var bodyArr = newBodies.ToArray();

            if (fixOverlapCheckBox.Checked)
                FixOverlaps(ref bodyArr, 3);

            BodyManager.Add(bodyArr);

            MainLoop.ResumePhysics();
        }

        private void AddBodiesToDisc(int count, int maxSize, int minSize, int bodyMass)
        {
            MainLoop.WaitForPause();

            var newBodies = new List<Body>();

            float px, py;
            float radius = float.Parse(OrbitRadiusTextBox.Text.Trim());
            float innerRadius = float.Parse(InOrbitRadiusTextBox.Text.Trim());

            Matter.Density = float.Parse(DensityTextBox.Text);
            var ellipse = new Ellipse(ViewportHelpers.ScreenPointToField(ViewportOffsets.ScreenCenter), radius);
            var inEllipse = new Ellipse(ViewportHelpers.ScreenPointToField(ViewportOffsets.ScreenCenter), innerRadius);

            for (int i = 0; i < count; i++)
            {
                var bodySize = Numbers.GetRandomFloat(minSize, maxSize);
                px = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                py = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);

                int its = 0;
                int maxIts = 100;
                bool outOfSpace = false;

                PointF newLoc = new PointF(px, py);

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
                    newLoc = new PointF(px, py);
                    its++;
                }

                if (outOfSpace)
                    continue;

                float newMass;
                bool byDist = layeredCheckBox.Checked;
                MatterType matter;

                if (byDist)
                {
                    var dist = newLoc.DistanceSqrt(ellipse.Location);
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

                float rndZ = Numbers.GetRandomFloat(-100, 100);
                newBodies.Add(BodyManager.NewBody(newLoc.X, newLoc.Y, rndZ, bodySize, newMass, color, int.Parse(LifeTimeTextBox.Text.Trim())));

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
