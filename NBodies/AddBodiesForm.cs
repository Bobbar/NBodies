using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Rendering;
using NBodies.Rules;

namespace NBodies
{
    public partial class AddBodiesForm : Form
    {
        private double _solarMass = 30000;

        public AddBodiesForm()
        {
            InitializeComponent();
            DensityTextBox.Text = Rules.Matter.Density.ToString();
        }

        private double CircleV(double rx, double ry, double centerMass)
        {
            double r2 = Math.Sqrt(rx * rx + ry * ry);
            double numerator = 0.000000977 * 1000000.0 * centerMass;
            return Math.Sqrt(numerator / r2);
        }


        private void AddBodiesToOrbit(int count, int maxSize, int minSize, int bodyMass, bool includeCenterMass, double centerMass)
        {
            MainLoop.WaitForPause();

            double px, py;
            float radius = float.Parse(OrbitRadiusTextBox.Text);
            Rules.Matter.Density = double.Parse(DensityTextBox.Text);
            centerMass *= Rules.Matter.Density * 2;

            var ellipse = new Ellipse(ScaleHelpers.ScaleMousePosRelative(RenderVars.ScreenCenter), radius);


            for (int i = 0; i < count; i++)
            {
                px = Numbers.GetRandomDouble(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                py = Numbers.GetRandomDouble(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);

                while (!PointHelper.PointInsideCircle(ellipse.Location, ellipse.Size, new PointF().FromDouble(px, py)))
                {
                    px = Numbers.GetRandomDouble(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                    py = Numbers.GetRandomDouble(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);
                }

                double magV = CircleV(px, py, centerMass);
                double absAngle = Math.Atan(Math.Abs(py / px));
                double thetaV = Math.PI / 2 - absAngle;
                double vx = -1 * Math.Sign(py) * Math.Cos(thetaV) * magV;
                double vy = Math.Sign(px) * Math.Sin(thetaV) * magV;

                var bodySize = Numbers.GetRandomDouble(minSize, maxSize);
                double newMass;

                if (bodyMass > 0)
                {
                    newMass = bodyMass;
                }
                else
                {
                    newMass = BodyManager.CalcMass(bodySize);
                }

                BodyManager.Add(px, py, vx, vy, bodySize, newMass, ColorHelper.RandomColor());
            }

            if (includeCenterMass)
            {
                BodyManager.Add(ellipse.Location, 15, centerMass, Color.Black, 1);
            }

            MainLoop.Resume();
        }

        private void AddBodiesToDisc(int count, int maxSize, int minSize, int bodyMass)
        {
            double px, py;
            float radius = float.Parse(OrbitRadiusTextBox.Text);
            Rules.Matter.Density = double.Parse(DensityTextBox.Text);
            var ellipse = new Ellipse(ScaleHelpers.ScaleMousePosRelative(RenderVars.ScreenCenter), radius);

            int nGas = (count / 8) * 7;
            int nMinerals = (count / 8);
            int bodyCount = 0;

            for (int i = 0; i < count; i++)
            {
                MatterType matter = Matter.Types[0];

                if (bodyCount <= nGas)
                {
                    matter = Matter.Types[Numbers.GetRandomInt(0, 1)];
                }
                else if (bodyCount >= nMinerals)
                {
                    matter = Matter.Types[Numbers.GetRandomInt(2, 4)];
                }


                bodyCount++;

                px = Numbers.GetRandomDouble(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                py = Numbers.GetRandomDouble(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);

                while (!PointHelper.PointInsideCircle(ellipse.Location, ellipse.Size, new PointF().FromDouble(px, py)))
                {
                    px = Numbers.GetRandomDouble(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                    py = Numbers.GetRandomDouble(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);
                }


                var bodySize = Numbers.GetRandomDouble(minSize, maxSize);
                double newMass;

                if (bodyMass > 0)
                {
                    newMass = bodyMass;
                }
                else
                {
                    newMass = BodyManager.CalcMass(bodySize, matter.Density);
                }

                BodyManager.Add(px, py, bodySize, newMass, matter.Color);
            }

        }



        private struct Ellipse
        {
            public PointF Location;
            public float Size;

            public Ellipse(PointF location, float size)
            {
                Location = location;
                Size = size;
            }
        }

        private void AddOrbitButton_Click(object sender, EventArgs e)
        {
            AddBodiesToOrbit(int.Parse(NumToAddTextBox.Text.Trim()), int.Parse(MaxSizeTextBox.Text.Trim()), int.Parse(MinSizeTextBox.Text.Trim()), int.Parse(MassTextBox.Text.Trim()), CenterMassCheckBox.Checked, double.Parse(CenterMassTextBox.Text.Trim()));
        }

        private void AddStationaryButton_Click(object sender, EventArgs e)
        {
            AddBodiesToDisc(int.Parse(NumToAddTextBox.Text.Trim()), int.Parse(MaxSizeTextBox.Text.Trim()), int.Parse(MinSizeTextBox.Text.Trim()), int.Parse(MassTextBox.Text.Trim()));
        }
    }
}
