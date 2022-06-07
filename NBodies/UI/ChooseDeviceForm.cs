using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.Physics;

namespace NBodies.UI
{
    public partial class ChooseDeviceForm : Form
    {
        public Cloo.ComputeDevice SelectedDevice;
        public int MaxThreadsPerBlock = 0;
        public bool FastMath = true;

        private List<Cloo.ComputeDevice> _devices = new List<Cloo.ComputeDevice>();


        public ChooseDeviceForm()
        {
            InitializeComponent();
            PopulateList();
        }

        private void PopulateList()
        {
            _devices = OpenCLPhysics.GetDevices();

            foreach (var dev in _devices)
            {
                deviceListBox.Items.Add($@"[{_devices.IndexOf(dev)}]  Name: {dev.Name}  Platform: {dev.Platform.Name}  Version: {dev.VersionString}");
            }
        }

        private void deviceListBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (deviceListBox.SelectedIndex >= 0)
            {
                SelectedDevice = _devices[deviceListBox.SelectedIndex];
                threadsTextBox.Text = SelectedDevice.MaxWorkGroupSize.ToString();
            }

        }

        private void OkButton_Click(object sender, EventArgs e)
        {
            int tpb = 0;
            
            if (int.TryParse(threadsTextBox.Text.Trim(), out tpb))
            {
                MaxThreadsPerBlock = tpb;
                FastMath = fastMathCheckBox.Checked;

                this.DialogResult = DialogResult.OK;
            }
        }
    }
}
