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
            this.tableLayoutPanel2 = new System.Windows.Forms.TableLayoutPanel();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.optionsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.antiAliasingToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.highContrastToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.saveStateToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.loadStateToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.tableLayoutPanel3 = new System.Windows.Forms.TableLayoutPanel();
            this.LoadRecordingButton = new System.Windows.Forms.Button();
            this.RecordButton = new System.Windows.Forms.Button();
            this.ScreenShotButton = new System.Windows.Forms.Button();
            this.RadiusTextBox = new System.Windows.Forms.TextBox();
            this.VeloYTextBox = new System.Windows.Forms.TextBox();
            this.VeloXTextBox = new System.Windows.Forms.TextBox();
            this.AddBodiesButton = new System.Windows.Forms.Button();
            this.RemoveAllButton = new System.Windows.Forms.Button();
            this.TimeStepUpDown = new System.Windows.Forms.NumericUpDown();
            this.FpsLimitTextBox = new System.Windows.Forms.TextBox();
            this.MassTextBox = new System.Windows.Forms.TextBox();
            this.FlagsTextBox = new System.Windows.Forms.TextBox();
            this.UpdateButton = new System.Windows.Forms.Button();
            this.TotalMassButton = new System.Windows.Forms.Button();
            this.TrailsCheckBox = new System.Windows.Forms.CheckBox();
            this.PauseButton = new System.Windows.Forms.CheckBox();
            this.panel1 = new System.Windows.Forms.Panel();
            this.FPSLabel = new System.Windows.Forms.Label();
            this.RenderBox = new System.Windows.Forms.PictureBox();
            this.BodyCountLabel = new System.Windows.Forms.Label();
            this.tableLayoutPanel1.SuspendLayout();
            this.tableLayoutPanel2.SuspendLayout();
            this.menuStrip1.SuspendLayout();
            this.tableLayoutPanel3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.TimeStepUpDown)).BeginInit();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.RenderBox)).BeginInit();
            this.SuspendLayout();
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 1;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.Controls.Add(this.tableLayoutPanel2, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.panel1, 0, 1);
            this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 0);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 2;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 63F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(1359, 706);
            this.tableLayoutPanel1.TabIndex = 0;
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
            this.tableLayoutPanel2.Size = new System.Drawing.Size(1353, 57);
            this.tableLayoutPanel2.TabIndex = 1;
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.optionsToolStripMenuItem,
            this.toolsToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(1353, 22);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // optionsToolStripMenuItem
            // 
            this.optionsToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.antiAliasingToolStripMenuItem,
            this.highContrastToolStripMenuItem});
            this.optionsToolStripMenuItem.Name = "optionsToolStripMenuItem";
            this.optionsToolStripMenuItem.Size = new System.Drawing.Size(61, 18);
            this.optionsToolStripMenuItem.Text = "Options";
            // 
            // antiAliasingToolStripMenuItem
            // 
            this.antiAliasingToolStripMenuItem.Checked = true;
            this.antiAliasingToolStripMenuItem.CheckOnClick = true;
            this.antiAliasingToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.antiAliasingToolStripMenuItem.Name = "antiAliasingToolStripMenuItem";
            this.antiAliasingToolStripMenuItem.Size = new System.Drawing.Size(148, 22);
            this.antiAliasingToolStripMenuItem.Text = "Anti-Aliasing";
            this.antiAliasingToolStripMenuItem.CheckedChanged += new System.EventHandler(this.antiAliasingToolStripMenuItem_CheckedChanged);
            // 
            // highContrastToolStripMenuItem
            // 
            this.highContrastToolStripMenuItem.CheckOnClick = true;
            this.highContrastToolStripMenuItem.Name = "highContrastToolStripMenuItem";
            this.highContrastToolStripMenuItem.Size = new System.Drawing.Size(148, 22);
            this.highContrastToolStripMenuItem.Text = "High Contrast";
            this.highContrastToolStripMenuItem.CheckedChanged += new System.EventHandler(this.highContrastToolStripMenuItem_CheckedChanged);
            // 
            // toolsToolStripMenuItem
            // 
            this.toolsToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.saveStateToolStripMenuItem,
            this.loadStateToolStripMenuItem});
            this.toolsToolStripMenuItem.Name = "toolsToolStripMenuItem";
            this.toolsToolStripMenuItem.Size = new System.Drawing.Size(47, 18);
            this.toolsToolStripMenuItem.Text = "Tools";
            // 
            // saveStateToolStripMenuItem
            // 
            this.saveStateToolStripMenuItem.Name = "saveStateToolStripMenuItem";
            this.saveStateToolStripMenuItem.Size = new System.Drawing.Size(129, 22);
            this.saveStateToolStripMenuItem.Text = "Save State";
            this.saveStateToolStripMenuItem.Click += new System.EventHandler(this.saveStateToolStripMenuItem_Click);
            // 
            // loadStateToolStripMenuItem
            // 
            this.loadStateToolStripMenuItem.Name = "loadStateToolStripMenuItem";
            this.loadStateToolStripMenuItem.Size = new System.Drawing.Size(129, 22);
            this.loadStateToolStripMenuItem.Text = "Load State";
            this.loadStateToolStripMenuItem.Click += new System.EventHandler(this.loadStateToolStripMenuItem_Click);
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
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 266F));
            this.tableLayoutPanel3.Controls.Add(this.LoadRecordingButton, 15, 0);
            this.tableLayoutPanel3.Controls.Add(this.RecordButton, 14, 0);
            this.tableLayoutPanel3.Controls.Add(this.ScreenShotButton, 13, 0);
            this.tableLayoutPanel3.Controls.Add(this.RadiusTextBox, 7, 0);
            this.tableLayoutPanel3.Controls.Add(this.VeloYTextBox, 6, 0);
            this.tableLayoutPanel3.Controls.Add(this.VeloXTextBox, 5, 0);
            this.tableLayoutPanel3.Controls.Add(this.AddBodiesButton, 0, 0);
            this.tableLayoutPanel3.Controls.Add(this.RemoveAllButton, 1, 0);
            this.tableLayoutPanel3.Controls.Add(this.TimeStepUpDown, 3, 0);
            this.tableLayoutPanel3.Controls.Add(this.FpsLimitTextBox, 4, 0);
            this.tableLayoutPanel3.Controls.Add(this.MassTextBox, 8, 0);
            this.tableLayoutPanel3.Controls.Add(this.FlagsTextBox, 9, 0);
            this.tableLayoutPanel3.Controls.Add(this.UpdateButton, 10, 0);
            this.tableLayoutPanel3.Controls.Add(this.TotalMassButton, 11, 0);
            this.tableLayoutPanel3.Controls.Add(this.TrailsCheckBox, 12, 0);
            this.tableLayoutPanel3.Controls.Add(this.PauseButton, 2, 0);
            this.tableLayoutPanel3.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel3.Location = new System.Drawing.Point(3, 25);
            this.tableLayoutPanel3.Name = "tableLayoutPanel3";
            this.tableLayoutPanel3.RowCount = 1;
            this.tableLayoutPanel3.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel3.Size = new System.Drawing.Size(1347, 29);
            this.tableLayoutPanel3.TabIndex = 1;
            // 
            // LoadRecordingButton
            // 
            this.LoadRecordingButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.LoadRecordingButton.Location = new System.Drawing.Point(1243, 3);
            this.LoadRecordingButton.Name = "LoadRecordingButton";
            this.LoadRecordingButton.Size = new System.Drawing.Size(260, 23);
            this.LoadRecordingButton.TabIndex = 15;
            this.LoadRecordingButton.Text = "Load Recording";
            this.LoadRecordingButton.UseVisualStyleBackColor = true;
            // 
            // RecordButton
            // 
            this.RecordButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RecordButton.Location = new System.Drawing.Point(1147, 3);
            this.RecordButton.Name = "RecordButton";
            this.RecordButton.Size = new System.Drawing.Size(90, 23);
            this.RecordButton.TabIndex = 14;
            this.RecordButton.Text = "Record";
            this.RecordButton.UseVisualStyleBackColor = true;
            // 
            // ScreenShotButton
            // 
            this.ScreenShotButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.ScreenShotButton.Location = new System.Drawing.Point(1063, 3);
            this.ScreenShotButton.Name = "ScreenShotButton";
            this.ScreenShotButton.Size = new System.Drawing.Size(78, 23);
            this.ScreenShotButton.TabIndex = 13;
            this.ScreenShotButton.Text = "Screenshot";
            this.ScreenShotButton.UseVisualStyleBackColor = true;
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
            this.AddBodiesButton.Size = new System.Drawing.Size(89, 23);
            this.AddBodiesButton.TabIndex = 0;
            this.AddBodiesButton.Text = "Add Bodies";
            this.AddBodiesButton.UseVisualStyleBackColor = true;
            this.AddBodiesButton.Click += new System.EventHandler(this.AddBodiesButton_Click);
            // 
            // RemoveAllButton
            // 
            this.RemoveAllButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RemoveAllButton.Location = new System.Drawing.Point(98, 3);
            this.RemoveAllButton.Name = "RemoveAllButton";
            this.RemoveAllButton.Size = new System.Drawing.Size(103, 23);
            this.RemoveAllButton.TabIndex = 1;
            this.RemoveAllButton.Text = "Remove All";
            this.RemoveAllButton.UseVisualStyleBackColor = true;
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
            this.UpdateButton.Size = new System.Drawing.Size(72, 23);
            this.UpdateButton.TabIndex = 10;
            this.UpdateButton.Text = "Update";
            this.UpdateButton.UseVisualStyleBackColor = true;
            // 
            // TotalMassButton
            // 
            this.TotalMassButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.TotalMassButton.Location = new System.Drawing.Point(917, 3);
            this.TotalMassButton.Name = "TotalMassButton";
            this.TotalMassButton.Size = new System.Drawing.Size(68, 23);
            this.TotalMassButton.TabIndex = 11;
            this.TotalMassButton.Text = "Tot. Mass";
            this.TotalMassButton.UseVisualStyleBackColor = true;
            // 
            // TrailsCheckBox
            // 
            this.TrailsCheckBox.Appearance = System.Windows.Forms.Appearance.Button;
            this.TrailsCheckBox.AutoSize = true;
            this.TrailsCheckBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.TrailsCheckBox.Location = new System.Drawing.Point(991, 3);
            this.TrailsCheckBox.Name = "TrailsCheckBox";
            this.TrailsCheckBox.Size = new System.Drawing.Size(66, 23);
            this.TrailsCheckBox.TabIndex = 16;
            this.TrailsCheckBox.Text = "Trails";
            this.TrailsCheckBox.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.TrailsCheckBox.UseVisualStyleBackColor = true;
            this.TrailsCheckBox.CheckedChanged += new System.EventHandler(this.TrailsCheckBox_CheckedChanged);
            // 
            // PauseButton
            // 
            this.PauseButton.Appearance = System.Windows.Forms.Appearance.Button;
            this.PauseButton.AutoSize = true;
            this.PauseButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.PauseButton.Location = new System.Drawing.Point(207, 3);
            this.PauseButton.Name = "PauseButton";
            this.PauseButton.Size = new System.Drawing.Size(91, 23);
            this.PauseButton.TabIndex = 17;
            this.PauseButton.Text = "Pause Physics";
            this.PauseButton.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.PauseButton.UseVisualStyleBackColor = true;
            this.PauseButton.CheckedChanged += new System.EventHandler(this.PauseButton_CheckedChanged);
            // 
            // panel1
            // 
            this.panel1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel1.Controls.Add(this.BodyCountLabel);
            this.panel1.Controls.Add(this.FPSLabel);
            this.panel1.Controls.Add(this.RenderBox);
            this.panel1.Location = new System.Drawing.Point(3, 66);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(1353, 637);
            this.panel1.TabIndex = 0;
            // 
            // FPSLabel
            // 
            this.FPSLabel.AutoSize = true;
            this.FPSLabel.BackColor = System.Drawing.Color.Black;
            this.FPSLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.FPSLabel.Location = new System.Drawing.Point(3, 14);
            this.FPSLabel.Name = "FPSLabel";
            this.FPSLabel.Size = new System.Drawing.Size(30, 13);
            this.FPSLabel.TabIndex = 1;
            this.FPSLabel.Text = "FPS:";
            // 
            // RenderBox
            // 
            this.RenderBox.BackColor = System.Drawing.Color.Black;
            this.RenderBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RenderBox.Location = new System.Drawing.Point(0, 0);
            this.RenderBox.Name = "RenderBox";
            this.RenderBox.Size = new System.Drawing.Size(1351, 635);
            this.RenderBox.TabIndex = 0;
            this.RenderBox.TabStop = false;
            this.RenderBox.MouseDown += new System.Windows.Forms.MouseEventHandler(this.RenderBox_MouseDown);
            this.RenderBox.MouseMove += new System.Windows.Forms.MouseEventHandler(this.RenderBox_MouseMove);
            this.RenderBox.MouseUp += new System.Windows.Forms.MouseEventHandler(this.RenderBox_MouseUp);
            this.RenderBox.Resize += new System.EventHandler(this.RenderBox_Resize);
            // 
            // BodyCountLabel
            // 
            this.BodyCountLabel.AutoSize = true;
            this.BodyCountLabel.BackColor = System.Drawing.Color.Black;
            this.BodyCountLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.BodyCountLabel.Location = new System.Drawing.Point(3, 27);
            this.BodyCountLabel.Name = "BodyCountLabel";
            this.BodyCountLabel.Size = new System.Drawing.Size(42, 13);
            this.BodyCountLabel.TabIndex = 2;
            this.BodyCountLabel.Text = "Bodies:";
            // 
            // DisplayForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1359, 706);
            this.Controls.Add(this.tableLayoutPanel1);
            this.DoubleBuffered = true;
            this.KeyPreview = true;
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "DisplayForm";
            this.Text = "NBodies";
            this.Load += new System.EventHandler(this.DisplayForm_Load);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.DisplayForm_KeyDown);
            this.KeyUp += new System.Windows.Forms.KeyEventHandler(this.DisplayForm_KeyUp);
            this.tableLayoutPanel1.ResumeLayout(false);
            this.tableLayoutPanel2.ResumeLayout(false);
            this.tableLayoutPanel2.PerformLayout();
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.tableLayoutPanel3.ResumeLayout(false);
            this.tableLayoutPanel3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.TimeStepUpDown)).EndInit();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.RenderBox)).EndInit();
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
        private System.Windows.Forms.NumericUpDown TimeStepUpDown;
        private System.Windows.Forms.TextBox FpsLimitTextBox;
        private System.Windows.Forms.TextBox MassTextBox;
        private System.Windows.Forms.TextBox FlagsTextBox;
        private System.Windows.Forms.Button UpdateButton;
        private System.Windows.Forms.Button TotalMassButton;
        private System.Windows.Forms.Button LoadRecordingButton;
        private System.Windows.Forms.Button RecordButton;
        private System.Windows.Forms.Button ScreenShotButton;
        private System.Windows.Forms.CheckBox TrailsCheckBox;
        private System.Windows.Forms.CheckBox PauseButton;
        private System.Windows.Forms.ToolStripMenuItem saveStateToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem loadStateToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem antiAliasingToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem highContrastToolStripMenuItem;
        private System.Windows.Forms.Label FPSLabel;
        private System.Windows.Forms.Label BodyCountLabel;
    }
}

