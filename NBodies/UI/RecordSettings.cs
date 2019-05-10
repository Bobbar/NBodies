using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NBodies.UI
{
    public partial class RecordSettings : Form
    {

        public float TimeStep
        {
            get { return _timeStep; }
        }

        public double MaxRecordSize
        {
            get { return _maxSize;  }
        }

        private float _timeStep;
        private double _maxSize;

        public RecordSettings()
        {
            InitializeComponent();

            _timeStep = MainLoop.RecordTimeStep;
            _maxSize = MainLoop.RecordMaxSize;

            TimeStepTextBox.Text = _timeStep.ToString();
            SizeLimitTextBox.Text = _maxSize.ToString();
        }

        private void OKButton_Click(object sender, EventArgs e)
        {

            if (float.TryParse(TimeStepTextBox.Text, out _timeStep))
            {
                if (_timeStep < 0 || _timeStep > 1.0f)
                    _timeStep = MainLoop.RecordTimeStep;
            }

            if (double.TryParse(SizeLimitTextBox.Text, out _maxSize))
            {
                if (_maxSize < 0)
                    _maxSize = 0;
            }

            DialogResult = DialogResult.OK;
        }
    }
}
