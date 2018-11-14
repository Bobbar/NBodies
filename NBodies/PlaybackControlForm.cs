using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.IO;

namespace NBodies
{
    public partial class PlaybackControlForm : Form
    {
        private IRecording _recorder;
        private bool _paused = false;

        public PlaybackControlForm()
        {
            InitializeComponent();
        }

        public PlaybackControlForm(IRecording recorder)
        {
            InitializeComponent();
            _recorder = recorder;
            _recorder.ProgressChanged += _recorder_ProgressChanged;
            SeekTrackBar.Maximum = _recorder.TotalFrames;
            SeekTrackBar.Value = 1;

            this.Show();
        }

        private void _recorder_ProgressChanged(object sender, int e)
        {
            if (SeekTrackBar.InvokeRequired)
            {
                SeekTrackBar.Invoke(new Action(() => _recorder_ProgressChanged(sender, e)));
            }
            else
            {
                if (e <= SeekTrackBar.Maximum)
                {
                    SeekTrackBar.Value = e;
                }
            }
        }

        private void PauseResumeButton_Click(object sender, EventArgs e)
        {
            _paused = !_paused;
            _recorder.PlaybackPaused = _paused;
        }

        private void SeekTrackBar_ValueChanged(object sender, EventArgs e)
        {
            if (_paused)
            {
                _recorder.SeekToFrame(SeekTrackBar.Value);
            }
        }

        private void SeekTrackBar_MouseDown(object sender, MouseEventArgs e)
        {
            _paused = true;
            _recorder.PlaybackPaused = true;
        }

        private void SeekTrackBar_MouseUp(object sender, MouseEventArgs e)
        {
            _paused = false;
            _recorder.PlaybackPaused = false;
        }
    }
}
