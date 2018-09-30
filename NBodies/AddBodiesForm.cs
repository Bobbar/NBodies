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

namespace NBodies
{
    public partial class AddBodiesForm : Form
    {
        private double _solarMass = 30000;
        private Random _rnd = new Random((int)(DateTime.Now.Ticks % Int32.MaxValue));

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

        private double GetRandomValue(double min, double max)
        {
            double range = max - min;
            double sample = _rnd.NextDouble();
            double scaled = (sample * range) + min;
            return scaled;
        }

        private void AddBodiesToOrbit(int count, int maxSize, int minSize, int bodyMass, bool includeCenterMass, double centerMass)
        {
            MainLoop.Pause();

            double px, py;
            float radius = float.Parse(OrbitRadiusTextBox.Text);
            Rules.Matter.Density = double.Parse(DensityTextBox.Text);
            centerMass *= Rules.Matter.Density * 2;

            var ellipse = new Ellipse(ScaleHelpers.ScaleMousePosRelative(RenderVars.ScreenCenter), radius);
            ellipse.Location.X += 1;
            ellipse.Location.Y += 1;

            for (int i = 0; i < count; i++)
            {
                px = GetRandomValue(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                py = GetRandomValue(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);

                while (!PointHelper.PointInsideCircle(ellipse.Location, ellipse.Size, new PointF().FromDouble(px, py)))
                {
                    px = GetRandomValue(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                    py = GetRandomValue(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);
                }

                double magV = CircleV(px, py, centerMass);
                double absAngle = Math.Atan(Math.Abs(py / px));
                double thetaV = Math.PI / 2 - absAngle;
                double vx = -1 * Math.Sign(py) * Math.Cos(thetaV) * magV;
                double vy = Math.Sign(px) * Math.Sin(thetaV) * magV;

                var bodySize = GetRandomValue(minSize, maxSize);
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
    }
}
