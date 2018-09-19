using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;


namespace NBodies
{
    public partial class DisplayForm : Form
    {
        private bool shiftDown = false;
        private bool ctrlDown = false;
        private long selectedIndex = -1;

        public DisplayForm()
        {
            InitializeComponent();
        }

       

        private void RenderBox_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                if (selectedIndex == -1 && shiftDown)
                {

                }
            }
        }
    }
}
