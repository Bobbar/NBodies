namespace NBodies
{
    partial class PlaybackControlForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.SeekTrackBar = new System.Windows.Forms.TrackBar();
            this.PauseResumeButton = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.SeekTrackBar)).BeginInit();
            this.SuspendLayout();
            // 
            // SeekTrackBar
            // 
            this.SeekTrackBar.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.SeekTrackBar.Location = new System.Drawing.Point(1, 12);
            this.SeekTrackBar.Name = "SeekTrackBar";
            this.SeekTrackBar.Size = new System.Drawing.Size(767, 45);
            this.SeekTrackBar.TabIndex = 0;
            this.SeekTrackBar.ValueChanged += new System.EventHandler(this.SeekTrackBar_ValueChanged);
            this.SeekTrackBar.MouseDown += new System.Windows.Forms.MouseEventHandler(this.SeekTrackBar_MouseDown);
            this.SeekTrackBar.MouseUp += new System.Windows.Forms.MouseEventHandler(this.SeekTrackBar_MouseUp);
            // 
            // PauseResumeButton
            // 
            this.PauseResumeButton.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.PauseResumeButton.Location = new System.Drawing.Point(303, 63);
            this.PauseResumeButton.Name = "PauseResumeButton";
            this.PauseResumeButton.Size = new System.Drawing.Size(163, 37);
            this.PauseResumeButton.TabIndex = 1;
            this.PauseResumeButton.Text = "Pause/Resume";
            this.PauseResumeButton.UseVisualStyleBackColor = true;
            this.PauseResumeButton.Click += new System.EventHandler(this.PauseResumeButton_Click);
            // 
            // PlaybackControlForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(769, 120);
            this.Controls.Add(this.PauseResumeButton);
            this.Controls.Add(this.SeekTrackBar);
            this.MaximumSize = new System.Drawing.Size(2000, 159);
            this.MinimizeBox = false;
            this.Name = "PlaybackControlForm";
            this.Opacity = 0.8D;
            this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Show;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Playback Control";
            ((System.ComponentModel.ISupportInitialize)(this.SeekTrackBar)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TrackBar SeekTrackBar;
        private System.Windows.Forms.Button PauseResumeButton;
    }
}