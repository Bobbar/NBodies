namespace NBodies
{
    partial class DisplayForm
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
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.panel1 = new System.Windows.Forms.Panel();
            this.RenderBox = new System.Windows.Forms.PictureBox();
            this.tableLayoutPanel2 = new System.Windows.Forms.TableLayoutPanel();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.optionsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.tableLayoutPanel3 = new System.Windows.Forms.TableLayoutPanel();
            this.LoadRecordingButton = new System.Windows.Forms.Button();
            this.RecordButton = new System.Windows.Forms.Button();
            this.ScreenShotButton = new System.Windows.Forms.Button();
            this.TrailsButton = new System.Windows.Forms.Button();
            this.RadiusTextBox = new System.Windows.Forms.TextBox();
            this.VeloYTextBox = new System.Windows.Forms.TextBox();
            this.VeloXTextBox = new System.Windows.Forms.TextBox();
            this.AddBodiesButton = new System.Windows.Forms.Button();
            this.RemoveAllButton = new System.Windows.Forms.Button();
            this.PauseButton = new System.Windows.Forms.Button();
            this.TimeStepUpDown = new System.Windows.Forms.NumericUpDown();
            this.FpsLimitTextBox = new System.Windows.Forms.TextBox();
            this.MassTextBox = new System.Windows.Forms.TextBox();
            this.FlagsTextBox = new System.Windows.Forms.TextBox();
            this.UpdateButton = new System.Windows.Forms.Button();
            this.TotalMassButton = new System.Windows.Forms.Button();
            this.tableLayoutPanel1.SuspendLayout();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.RenderBox)).BeginInit();
            this.tableLayoutPanel2.SuspendLayout();
            this.menuStrip1.SuspendLayout();
            this.tableLayoutPanel3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.TimeStepUpDown)).BeginInit();
            this.SuspendLayout();
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 1;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.Controls.Add(this.panel1, 0, 1);
            this.tableLayoutPanel1.Controls.Add(this.tableLayoutPanel2, 0, 0);
            this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 0);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 2;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 7.692307F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 92.30769F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(1429, 819);
            this.tableLayoutPanel1.TabIndex = 0;
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.RenderBox);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel1.Location = new System.Drawing.Point(3, 65);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(1423, 751);
            this.panel1.TabIndex = 0;
            // 
            // RenderBox
            // 
            this.RenderBox.BackColor = System.Drawing.Color.Black;
            this.RenderBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RenderBox.Location = new System.Drawing.Point(0, 0);
            this.RenderBox.Name = "RenderBox";
            this.RenderBox.Size = new System.Drawing.Size(1423, 751);
            this.RenderBox.TabIndex = 0;
            this.RenderBox.TabStop = false;
            this.RenderBox.MouseDown += new System.Windows.Forms.MouseEventHandler(this.RenderBox_MouseDown);
            this.RenderBox.MouseMove += new System.Windows.Forms.MouseEventHandler(this.RenderBox_MouseMove);
            this.RenderBox.MouseUp += new System.Windows.Forms.MouseEventHandler(this.RenderBox_MouseUp);
            // 
            // tableLayoutPanel2
            // 
            this.tableLayoutPanel2.ColumnCount = 1;
            this.tableLayoutPanel2.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel2.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 20F));
            this.tableLayoutPanel2.Controls.Add(this.menuStrip1, 0, 0);
            this.tableLayoutPanel2.Controls.Add(this.tableLayoutPanel3, 0, 1);
            this.tableLayoutPanel2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel2.Location = new System.Drawing.Point(3, 3);
            this.tableLayoutPanel2.Name = "tableLayoutPanel2";
            this.tableLayoutPanel2.RowCount = 2;
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 39.43662F));
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 60.56338F));
            this.tableLayoutPanel2.Size = new System.Drawing.Size(1423, 56);
            this.tableLayoutPanel2.TabIndex = 1;
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.optionsToolStripMenuItem,
            this.toolsToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(1423, 22);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // optionsToolStripMenuItem
            // 
            this.optionsToolStripMenuItem.Name = "optionsToolStripMenuItem";
            this.optionsToolStripMenuItem.Size = new System.Drawing.Size(61, 18);
            this.optionsToolStripMenuItem.Text = "Options";
            // 
            // toolsToolStripMenuItem
            // 
            this.toolsToolStripMenuItem.Name = "toolsToolStripMenuItem";
            this.toolsToolStripMenuItem.Size = new System.Drawing.Size(47, 18);
            this.toolsToolStripMenuItem.Text = "Tools";
            // 
            // tableLayoutPanel3
            // 
            this.tableLayoutPanel3.ColumnCount = 16;
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 95F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 109F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 97F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 68F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 40F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 98F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 96F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 96F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 89F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 48F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 78F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 74F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 72F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 84F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 96F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 177F));
            this.tableLayoutPanel3.Controls.Add(this.LoadRecordingButton, 15, 0);
            this.tableLayoutPanel3.Controls.Add(this.RecordButton, 14, 0);
            this.tableLayoutPanel3.Controls.Add(this.ScreenShotButton, 13, 0);
            this.tableLayoutPanel3.Controls.Add(this.TrailsButton, 12, 0);
            this.tableLayoutPanel3.Controls.Add(this.RadiusTextBox, 7, 0);
            this.tableLayoutPanel3.Controls.Add(this.VeloYTextBox, 6, 0);
            this.tableLayoutPanel3.Controls.Add(this.VeloXTextBox, 5, 0);
            this.tableLayoutPanel3.Controls.Add(this.AddBodiesButton, 0, 0);
            this.tableLayoutPanel3.Controls.Add(this.RemoveAllButton, 1, 0);
            this.tableLayoutPanel3.Controls.Add(this.PauseButton, 2, 0);
            this.tableLayoutPanel3.Controls.Add(this.TimeStepUpDown, 3, 0);
            this.tableLayoutPanel3.Controls.Add(this.FpsLimitTextBox, 4, 0);
            this.tableLayoutPanel3.Controls.Add(this.MassTextBox, 8, 0);
            this.tableLayoutPanel3.Controls.Add(this.FlagsTextBox, 9, 0);
            this.tableLayoutPanel3.Controls.Add(this.UpdateButton, 10, 0);
            this.tableLayoutPanel3.Controls.Add(this.TotalMassButton, 11, 0);
            this.tableLayoutPanel3.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel3.Location = new System.Drawing.Point(3, 25);
            this.tableLayoutPanel3.Name = "tableLayoutPanel3";
            this.tableLayoutPanel3.RowCount = 1;
            this.tableLayoutPanel3.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel3.Size = new System.Drawing.Size(1417, 28);
            this.tableLayoutPanel3.TabIndex = 1;
            // 
            // LoadRecordingButton
            // 
            this.LoadRecordingButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.LoadRecordingButton.Location = new System.Drawing.Point(1243, 3);
            this.LoadRecordingButton.Name = "LoadRecordingButton";
            this.LoadRecordingButton.Size = new System.Drawing.Size(171, 22);
            this.LoadRecordingButton.TabIndex = 15;
            this.LoadRecordingButton.Text = "Load Recording";
            this.LoadRecordingButton.UseVisualStyleBackColor = true;
            // 
            // RecordButton
            // 
            this.RecordButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RecordButton.Location = new System.Drawing.Point(1147, 3);
            this.RecordButton.Name = "RecordButton";
            this.RecordButton.Size = new System.Drawing.Size(90, 22);
            this.RecordButton.TabIndex = 14;
            this.RecordButton.Text = "Record";
            this.RecordButton.UseVisualStyleBackColor = true;
            // 
            // ScreenShotButton
            // 
            this.ScreenShotButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.ScreenShotButton.Location = new System.Drawing.Point(1063, 3);
            this.ScreenShotButton.Name = "ScreenShotButton";
            this.ScreenShotButton.Size = new System.Drawing.Size(78, 22);
            this.ScreenShotButton.TabIndex = 13;
            this.ScreenShotButton.Text = "Screenshot";
            this.ScreenShotButton.UseVisualStyleBackColor = true;
            // 
            // TrailsButton
            // 
            this.TrailsButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.TrailsButton.Location = new System.Drawing.Point(991, 3);
            this.TrailsButton.Name = "TrailsButton";
            this.TrailsButton.Size = new System.Drawing.Size(66, 22);
            this.TrailsButton.TabIndex = 12;
            this.TrailsButton.Text = "Trails";
            this.TrailsButton.UseVisualStyleBackColor = true;
            // 
            // RadiusTextBox
            // 
            this.RadiusTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RadiusTextBox.Location = new System.Drawing.Point(606, 3);
            this.RadiusTextBox.Name = "RadiusTextBox";
            this.RadiusTextBox.Size = new System.Drawing.Size(90, 20);
            this.RadiusTextBox.TabIndex = 7;
            // 
            // VeloYTextBox
            // 
            this.VeloYTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.VeloYTextBox.Location = new System.Drawing.Point(510, 3);
            this.VeloYTextBox.Name = "VeloYTextBox";
            this.VeloYTextBox.Size = new System.Drawing.Size(90, 20);
            this.VeloYTextBox.TabIndex = 6;
            // 
            // VeloXTextBox
            // 
            this.VeloXTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.VeloXTextBox.Location = new System.Drawing.Point(412, 3);
            this.VeloXTextBox.Name = "VeloXTextBox";
            this.VeloXTextBox.Size = new System.Drawing.Size(92, 20);
            this.VeloXTextBox.TabIndex = 5;
            // 
            // AddBodiesButton
            // 
            this.AddBodiesButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.AddBodiesButton.Location = new System.Drawing.Point(3, 3);
            this.AddBodiesButton.Name = "AddBodiesButton";
            this.AddBodiesButton.Size = new System.Drawing.Size(89, 22);
            this.AddBodiesButton.TabIndex = 0;
            this.AddBodiesButton.Text = "Add Bodies";
            this.AddBodiesButton.UseVisualStyleBackColor = true;
            // 
            // RemoveAllButton
            // 
            this.RemoveAllButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RemoveAllButton.Location = new System.Drawing.Point(98, 3);
            this.RemoveAllButton.Name = "RemoveAllButton";
            this.RemoveAllButton.Size = new System.Drawing.Size(103, 22);
            this.RemoveAllButton.TabIndex = 1;
            this.RemoveAllButton.Text = "Remove All";
            this.RemoveAllButton.UseVisualStyleBackColor = true;
            // 
            // PauseButton
            // 
            this.PauseButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.PauseButton.Location = new System.Drawing.Point(207, 3);
            this.PauseButton.Name = "PauseButton";
            this.PauseButton.Size = new System.Drawing.Size(91, 22);
            this.PauseButton.TabIndex = 2;
            this.PauseButton.Text = "Pause";
            this.PauseButton.UseVisualStyleBackColor = true;
            // 
            // TimeStepUpDown
            // 
            this.TimeStepUpDown.Dock = System.Windows.Forms.DockStyle.Fill;
            this.TimeStepUpDown.Location = new System.Drawing.Point(304, 3);
            this.TimeStepUpDown.Name = "TimeStepUpDown";
            this.TimeStepUpDown.Size = new System.Drawing.Size(62, 20);
            this.TimeStepUpDown.TabIndex = 3;
            // 
            // FpsLimitTextBox
            // 
            this.FpsLimitTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.FpsLimitTextBox.Location = new System.Drawing.Point(372, 3);
            this.FpsLimitTextBox.Name = "FpsLimitTextBox";
            this.FpsLimitTextBox.Size = new System.Drawing.Size(34, 20);
            this.FpsLimitTextBox.TabIndex = 4;
            // 
            // MassTextBox
            // 
            this.MassTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.MassTextBox.Location = new System.Drawing.Point(702, 3);
            this.MassTextBox.Name = "MassTextBox";
            this.MassTextBox.Size = new System.Drawing.Size(83, 20);
            this.MassTextBox.TabIndex = 8;
            // 
            // FlagsTextBox
            // 
            this.FlagsTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.FlagsTextBox.Location = new System.Drawing.Point(791, 3);
            this.FlagsTextBox.Name = "FlagsTextBox";
            this.FlagsTextBox.Size = new System.Drawing.Size(42, 20);
            this.FlagsTextBox.TabIndex = 9;
            // 
            // UpdateButton
            // 
            this.UpdateButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.UpdateButton.Location = new System.Drawing.Point(839, 3);
            this.UpdateButton.Name = "UpdateButton";
            this.UpdateButton.Size = new System.Drawing.Size(72, 22);
            this.UpdateButton.TabIndex = 10;
            this.UpdateButton.Text = "Update";
            this.UpdateButton.UseVisualStyleBackColor = true;
            // 
            // TotalMassButton
            // 
            this.TotalMassButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.TotalMassButton.Location = new System.Drawing.Point(917, 3);
            this.TotalMassButton.Name = "TotalMassButton";
            this.TotalMassButton.Size = new System.Drawing.Size(68, 22);
            this.TotalMassButton.TabIndex = 11;
            this.TotalMassButton.Text = "Tot. Mass";
            this.TotalMassButton.UseVisualStyleBackColor = true;
            // 
            // DisplayForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1429, 819);
            this.Controls.Add(this.tableLayoutPanel1);
            this.DoubleBuffered = true;
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "DisplayForm";
            this.Text = "NBodies";
            this.Load += new System.EventHandler(this.DisplayForm_Load);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.DisplayForm_KeyDown);
            this.KeyUp += new System.Windows.Forms.KeyEventHandler(this.DisplayForm_KeyUp);
            this.tableLayoutPanel1.ResumeLayout(false);
            this.panel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.RenderBox)).EndInit();
            this.tableLayoutPanel2.ResumeLayout(false);
            this.tableLayoutPanel2.PerformLayout();
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.tableLayoutPanel3.ResumeLayout(false);
            this.tableLayoutPanel3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.TimeStepUpDown)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.PictureBox RenderBox;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel2;
        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem optionsToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem toolsToolStripMenuItem;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel3;
        private System.Windows.Forms.TextBox RadiusTextBox;
        private System.Windows.Forms.TextBox VeloYTextBox;
        private System.Windows.Forms.TextBox VeloXTextBox;
        private System.Windows.Forms.Button AddBodiesButton;
        private System.Windows.Forms.Button RemoveAllButton;
        private System.Windows.Forms.Button PauseButton;
        private System.Windows.Forms.NumericUpDown TimeStepUpDown;
        private System.Windows.Forms.TextBox FpsLimitTextBox;
        private System.Windows.Forms.TextBox MassTextBox;
        private System.Windows.Forms.TextBox FlagsTextBox;
        private System.Windows.Forms.Button UpdateButton;
        private System.Windows.Forms.Button TotalMassButton;
        private System.Windows.Forms.Button LoadRecordingButton;
        private System.Windows.Forms.Button RecordButton;
        private System.Windows.Forms.Button ScreenShotButton;
        private System.Windows.Forms.Button TrailsButton;
    }
}

