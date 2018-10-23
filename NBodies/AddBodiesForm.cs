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
using NBodies.Shapes;

namespace NBodies
{
    public partial class AddBodiesForm : Form
    {
        private float _solarMass = 30000;

        public AddBodiesForm()
        {
            InitializeComponent();
            DensityTextBox.Text = Rules.Matter.Density.ToString();
        }

        private float CircleV(float rx, float ry, float centerMass)
        {
            float r2 = (float)Math.Sqrt(rx * rx + ry * ry);
            float numerator = (float)(0.000000977 * 1000000.0 * centerMass);
            return (float)Math.Sqrt(numerator / r2);
        }


        private void AddBodiesToOrbit(int count, int maxSize, int minSize, int bodyMass, bool includeCenterMass, float centerMass)
        {
            MainLoop.WaitForPause();

            float px, py;
            float radius = float.Parse(OrbitRadiusTextBox.Text);
            Rules.Matter.Density = float.Parse(DensityTextBox.Text);
            centerMass *= Rules.Matter.Density * 2;

            int nGas = (count / 8) * 7;
            int nMinerals = (count / 8);
            int bodyCount = 0;

            var ellipse = new Ellipse(ScaleHelpers.ScalePointRelative(RenderVars.ScreenCenter), radius);

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

                px = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                py = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);

                while (!PointHelper.PointInsideCircle(ellipse.Location, ellipse.Size, new PointF(px, py)))
                {
                    px = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                    py = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);
                }

                float magV = CircleV(px, py, centerMass);
                float absAngle = (float)Math.Atan(Math.Abs(py / px));
                float thetaV = (float)Math.PI * 0.5f - absAngle;
                float vx = -1 * (float)(Math.Sign(py) * Math.Cos(thetaV) * magV);
                float vy = (float)(Math.Sign(px) * Math.Sin(thetaV) * magV);

                var bodySize = Numbers.GetRandomFloat(minSize, maxSize);
                float newMass = 1;

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

                BodyManager.Add(px, py, vx, vy, bodySize, newMass, (StaticDensityCheckBox.Checked ? ColorHelper.RandomColor() : matter.Color));
            }

            if (includeCenterMass)
            {
                BodyManager.Add(ellipse.Location, 15, centerMass, Color.Black, 1);
            }

            MainLoop.Resume();
        }

        private void AddBodiesToDisc(int count, int maxSize, int minSize, int bodyMass)
        {
            MainLoop.WaitForPause();


            float px, py;
            float radius = float.Parse(OrbitRadiusTextBox.Text);
            Rules.Matter.Density = float.Parse(DensityTextBox.Text);
            var ellipse = new Ellipse(ScaleHelpers.ScalePointRelative(RenderVars.ScreenCenter), radius);

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

                px = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                py = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);

                while (!PointHelper.PointInsideCircle(ellipse.Location, ellipse.Size, new PointF(px, py)))
                {
                    px = Numbers.GetRandomFloat(ellipse.Location.X - ellipse.Size, ellipse.Location.X + ellipse.Size);
                    py = Numbers.GetRandomFloat(ellipse.Location.Y - ellipse.Size, ellipse.Location.Y + ellipse.Size);
                }


                var bodySize = Numbers.GetRandomFloat(minSize, maxSize);
                float newMass;


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
              
                BodyManager.Add(px, py, bodySize, newMass, (StaticDensityCheckBox.Checked ? ColorHelper.RandomColor() : matter.Color));
            }

            MainLoop.Resume();


        }





        private void AddOrbitButton_Click(object sender, EventArgs e)
        {
            AddBodiesToOrbit(int.Parse(NumToAddTextBox.Text.Trim()), int.Parse(MaxSizeTextBox.Text.Trim()), int.Parse(MinSizeTextBox.Text.Trim()), int.Parse(MassTextBox.Text.Trim()), CenterMassCheckBox.Checked, float.Parse(CenterMassTextBox.Text.Trim()));
        }

        private void AddStationaryButton_Click(object sender, EventArgs e)
        {
            AddBodiesToDisc(int.Parse(NumToAddTextBox.Text.Trim()), int.Parse(MaxSizeTextBox.Text.Trim()), int.Parse(MinSizeTextBox.Text.Trim()), int.Parse(MassTextBox.Text.Trim()));
        }
    }
}
