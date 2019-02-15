namespace NBodies.UI
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
            this.RootLayoutTable = new System.Windows.Forms.TableLayoutPanel();
            this.tableLayoutPanel2 = new System.Windows.Forms.TableLayoutPanel();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.optionsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.drawToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.antiAliasingToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.clipToViewportToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.rocheLimitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.showMeshToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.allForceVectorsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.displayToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.normalToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.pressuresToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.speedsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.indexToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.forcesToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.highContrastToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.followingBodyToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.showFollowBodyForce = new System.Windows.Forms.ToolStripMenuItem();
            this.showPredictOrbit = new System.Windows.Forms.ToolStripMenuItem();
            this.toolsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.saveStateToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.loadStateToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.reloadPreviousToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.tableLayoutPanel3 = new System.Windows.Forms.TableLayoutPanel();
            this.LoadRecordingButton = new System.Windows.Forms.Button();
            this.RecordButton = new System.Windows.Forms.Button();
            this.ToggleRendererButton = new System.Windows.Forms.Button();
            this.RadiusTextBox = new System.Windows.Forms.TextBox();
            this.VeloYTextBox = new System.Windows.Forms.TextBox();
            this.VeloXTextBox = new System.Windows.Forms.TextBox();
            this.AddBodiesButton = new System.Windows.Forms.Button();
            this.RemoveAllButton = new System.Windows.Forms.Button();
            this.TimeStepUpDown = new System.Windows.Forms.NumericUpDown();
            this.MassTextBox = new System.Windows.Forms.TextBox();
            this.FlagsTextBox = new System.Windows.Forms.TextBox();
            this.UpdateButton = new System.Windows.Forms.Button();
            this.CenterOnMassButton = new System.Windows.Forms.Button();
            this.TrailsCheckBox = new System.Windows.Forms.CheckBox();
            this.PauseButton = new System.Windows.Forms.CheckBox();
            this.PressureScaleUpDown = new System.Windows.Forms.NumericUpDown();
            this.AlphaUpDown = new System.Windows.Forms.NumericUpDown();
            this.panel1 = new System.Windows.Forms.Panel();
            this.RendererLabel = new System.Windows.Forms.Label();
            this.ScaleLabel = new System.Windows.Forms.Label();
            this.RecSizeLabel = new System.Windows.Forms.Label();
            this.SpeedLabel = new System.Windows.Forms.Label();
            this.FrameCountLabel = new System.Windows.Forms.Label();
            this.PressureLabel = new System.Windows.Forms.Label();
            this.DensityLabel = new System.Windows.Forms.Label();
            this.TotalMassLabel = new System.Windows.Forms.Label();
            this.BodyCountLabel = new System.Windows.Forms.Label();
            this.FPSLabel = new System.Windows.Forms.Label();
            this.RenderBox = new System.Windows.Forms.PictureBox();
            this.RootLayoutTable.SuspendLayout();
            this.tableLayoutPanel2.SuspendLayout();
            this.menuStrip1.SuspendLayout();
            this.tableLayoutPanel3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.TimeStepUpDown)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.PressureScaleUpDown)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.AlphaUpDown)).BeginInit();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.RenderBox)).BeginInit();
            this.SuspendLayout();
            // 
            // RootLayoutTable
            // 
            this.RootLayoutTable.ColumnCount = 1;
            this.RootLayoutTable.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.RootLayoutTable.Controls.Add(this.tableLayoutPanel2, 0, 0);
            this.RootLayoutTable.Controls.Add(this.panel1, 0, 1);
            this.RootLayoutTable.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RootLayoutTable.Location = new System.Drawing.Point(0, 0);
            this.RootLayoutTable.Name = "RootLayoutTable";
            this.RootLayoutTable.RowCount = 2;
            this.RootLayoutTable.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 63F));
            this.RootLayoutTable.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.RootLayoutTable.Size = new System.Drawing.Size(1589, 815);
            this.RootLayoutTable.TabIndex = 0;
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
            this.tableLayoutPanel2.Size = new System.Drawing.Size(1583, 57);
            this.tableLayoutPanel2.TabIndex = 1;
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.optionsToolStripMenuItem,
            this.toolsToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(1583, 22);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // optionsToolStripMenuItem
            // 
            this.optionsToolStripMenuItem.CheckOnClick = true;
            this.optionsToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.drawToolStripMenuItem,
            this.antiAliasingToolStripMenuItem,
            this.clipToViewportToolStripMenuItem,
            this.rocheLimitToolStripMenuItem,
            this.showMeshToolStripMenuItem,
            this.allForceVectorsToolStripMenuItem,
            this.displayToolStripMenuItem,
            this.followingBodyToolStripMenuItem});
            this.optionsToolStripMenuItem.Name = "optionsToolStripMenuItem";
            this.optionsToolStripMenuItem.Size = new System.Drawing.Size(61, 18);
            this.optionsToolStripMenuItem.Text = "Options";
            // 
            // drawToolStripMenuItem
            // 
            this.drawToolStripMenuItem.Checked = true;
            this.drawToolStripMenuItem.CheckOnClick = true;
            this.drawToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.drawToolStripMenuItem.Name = "drawToolStripMenuItem";
            this.drawToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.drawToolStripMenuItem.Text = "Draw";
            this.drawToolStripMenuItem.CheckedChanged += new System.EventHandler(this.drawToolStripMenuItem_CheckedChanged);
            // 
            // antiAliasingToolStripMenuItem
            // 
            this.antiAliasingToolStripMenuItem.Checked = true;
            this.antiAliasingToolStripMenuItem.CheckOnClick = true;
            this.antiAliasingToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.antiAliasingToolStripMenuItem.Name = "antiAliasingToolStripMenuItem";
            this.antiAliasingToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.antiAliasingToolStripMenuItem.Text = "Anti-Aliasing";
            this.antiAliasingToolStripMenuItem.CheckedChanged += new System.EventHandler(this.antiAliasingToolStripMenuItem_CheckedChanged);
            // 
            // clipToViewportToolStripMenuItem
            // 
            this.clipToViewportToolStripMenuItem.Checked = true;
            this.clipToViewportToolStripMenuItem.CheckOnClick = true;
            this.clipToViewportToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.clipToViewportToolStripMenuItem.Name = "clipToViewportToolStripMenuItem";
            this.clipToViewportToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.clipToViewportToolStripMenuItem.Text = "Clip To Viewport";
            this.clipToViewportToolStripMenuItem.CheckedChanged += new System.EventHandler(this.clipToViewportToolStripMenuItem_CheckedChanged);
            // 
            // rocheLimitToolStripMenuItem
            // 
            this.rocheLimitToolStripMenuItem.Checked = true;
            this.rocheLimitToolStripMenuItem.CheckOnClick = true;
            this.rocheLimitToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.rocheLimitToolStripMenuItem.Name = "rocheLimitToolStripMenuItem";
            this.rocheLimitToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.rocheLimitToolStripMenuItem.Text = "Roche Limit";
            this.rocheLimitToolStripMenuItem.CheckedChanged += new System.EventHandler(this.rocheLimitToolStripMenuItem_CheckedChanged);
            // 
            // showMeshToolStripMenuItem
            // 
            this.showMeshToolStripMenuItem.CheckOnClick = true;
            this.showMeshToolStripMenuItem.Name = "showMeshToolStripMenuItem";
            this.showMeshToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.showMeshToolStripMenuItem.Text = "Show Mesh";
            this.showMeshToolStripMenuItem.CheckedChanged += new System.EventHandler(this.showMeshToolStripMenuItem_CheckedChanged);
            // 
            // allForceVectorsToolStripMenuItem
            // 
            this.allForceVectorsToolStripMenuItem.CheckOnClick = true;
            this.allForceVectorsToolStripMenuItem.Name = "allForceVectorsToolStripMenuItem";
            this.allForceVectorsToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.allForceVectorsToolStripMenuItem.Text = "All Force Vectors";
            this.allForceVectorsToolStripMenuItem.CheckedChanged += new System.EventHandler(this.allForceVectorsToolStripMenuItem_CheckedChanged);
            // 
            // displayToolStripMenuItem
            // 
            this.displayToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.normalToolStripMenuItem,
            this.pressuresToolStripMenuItem,
            this.speedsToolStripMenuItem,
            this.indexToolStripMenuItem,
            this.forcesToolStripMenuItem,
            this.highContrastToolStripMenuItem});
            this.displayToolStripMenuItem.Name = "displayToolStripMenuItem";
            this.displayToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.displayToolStripMenuItem.Text = "Display";
            // 
            // normalToolStripMenuItem
            // 
            this.normalToolStripMenuItem.Checked = true;
            this.normalToolStripMenuItem.CheckOnClick = true;
            this.normalToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.normalToolStripMenuItem.Name = "normalToolStripMenuItem";
            this.normalToolStripMenuItem.Size = new System.Drawing.Size(148, 22);
            this.normalToolStripMenuItem.Text = "Normal";
            this.normalToolStripMenuItem.Click += new System.EventHandler(this.normalToolStripMenuItem_Click);
            // 
            // pressuresToolStripMenuItem
            // 
            this.pressuresToolStripMenuItem.CheckOnClick = true;
            this.pressuresToolStripMenuItem.Name = "pressuresToolStripMenuItem";
            this.pressuresToolStripMenuItem.Size = new System.Drawing.Size(148, 22);
            this.pressuresToolStripMenuItem.Text = "Pressures";
            this.pressuresToolStripMenuItem.Click += new System.EventHandler(this.pressuresToolStripMenuItem_Click);
            // 
            // speedsToolStripMenuItem
            // 
            this.speedsToolStripMenuItem.CheckOnClick = true;
            this.speedsToolStripMenuItem.Name = "speedsToolStripMenuItem";
            this.speedsToolStripMenuItem.Size = new System.Drawing.Size(148, 22);
            this.speedsToolStripMenuItem.Text = "Speeds";
            this.speedsToolStripMenuItem.Click += new System.EventHandler(this.speedsToolStripMenuItem_Click);
            // 
            // indexToolStripMenuItem
            // 
            this.indexToolStripMenuItem.CheckOnClick = true;
            this.indexToolStripMenuItem.Name = "indexToolStripMenuItem";
            this.indexToolStripMenuItem.Size = new System.Drawing.Size(148, 22);
            this.indexToolStripMenuItem.Text = "Index";
            this.indexToolStripMenuItem.Click += new System.EventHandler(this.indexToolStripMenuItem_Click);
            // 
            // forcesToolStripMenuItem
            // 
            this.forcesToolStripMenuItem.CheckOnClick = true;
            this.forcesToolStripMenuItem.Name = "forcesToolStripMenuItem";
            this.forcesToolStripMenuItem.Size = new System.Drawing.Size(148, 22);
            this.forcesToolStripMenuItem.Text = "Forces";
            this.forcesToolStripMenuItem.Click += new System.EventHandler(this.forcesToolStripMenuItem_Click);
            // 
            // highContrastToolStripMenuItem
            // 
            this.highContrastToolStripMenuItem.CheckOnClick = true;
            this.highContrastToolStripMenuItem.Name = "highContrastToolStripMenuItem";
            this.highContrastToolStripMenuItem.Size = new System.Drawing.Size(148, 22);
            this.highContrastToolStripMenuItem.Text = "High Contrast";
            this.highContrastToolStripMenuItem.Click += new System.EventHandler(this.highContrastToolStripMenuItem1_Click);
            // 
            // followingBodyToolStripMenuItem
            // 
            this.followingBodyToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.showFollowBodyForce,
            this.showPredictOrbit});
            this.followingBodyToolStripMenuItem.Name = "followingBodyToolStripMenuItem";
            this.followingBodyToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.followingBodyToolStripMenuItem.Text = "Follow Body Display";
            // 
            // showFollowBodyForce
            // 
            this.showFollowBodyForce.CheckOnClick = true;
            this.showFollowBodyForce.Name = "showFollowBodyForce";
            this.showFollowBodyForce.Size = new System.Drawing.Size(139, 22);
            this.showFollowBodyForce.Text = "Force Vector";
            this.showFollowBodyForce.CheckedChanged += new System.EventHandler(this.showFollowBodyForce_CheckedChanged);
            // 
            // showPredictOrbit
            // 
            this.showPredictOrbit.CheckOnClick = true;
            this.showPredictOrbit.Name = "showPredictOrbit";
            this.showPredictOrbit.Size = new System.Drawing.Size(139, 22);
            this.showPredictOrbit.Text = "Orbit (Slow)";
            this.showPredictOrbit.CheckedChanged += new System.EventHandler(this.showPredictOrbit_CheckedChanged);
            // 
            // toolsToolStripMenuItem
            // 
            this.toolsToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.saveStateToolStripMenuItem,
            this.loadStateToolStripMenuItem,
            this.reloadPreviousToolStripMenuItem});
            this.toolsToolStripMenuItem.Name = "toolsToolStripMenuItem";
            this.toolsToolStripMenuItem.Size = new System.Drawing.Size(47, 18);
            this.toolsToolStripMenuItem.Text = "Tools";
            // 
            // saveStateToolStripMenuItem
            // 
            this.saveStateToolStripMenuItem.Name = "saveStateToolStripMenuItem";
            this.saveStateToolStripMenuItem.Size = new System.Drawing.Size(158, 22);
            this.saveStateToolStripMenuItem.Text = "Save State";
            this.saveStateToolStripMenuItem.Click += new System.EventHandler(this.saveStateToolStripMenuItem_Click);
            // 
            // loadStateToolStripMenuItem
            // 
            this.loadStateToolStripMenuItem.Name = "loadStateToolStripMenuItem";
            this.loadStateToolStripMenuItem.Size = new System.Drawing.Size(158, 22);
            this.loadStateToolStripMenuItem.Text = "Load State";
            this.loadStateToolStripMenuItem.Click += new System.EventHandler(this.loadStateToolStripMenuItem_Click);
            // 
            // reloadPreviousToolStripMenuItem
            // 
            this.reloadPreviousToolStripMenuItem.Name = "reloadPreviousToolStripMenuItem";
            this.reloadPreviousToolStripMenuItem.Size = new System.Drawing.Size(158, 22);
            this.reloadPreviousToolStripMenuItem.Text = "Reload Previous";
            this.reloadPreviousToolStripMenuItem.Click += new System.EventHandler(this.reloadPreviousToolStripMenuItem_Click);
            // 
            // tableLayoutPanel3
            // 
            this.tableLayoutPanel3.ColumnCount = 17;
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 95F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 109F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 97F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 68F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 48F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 47F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 96F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 101F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 49F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 59F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 40F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 81F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 95F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 81F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 107F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 110F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 294F));
            this.tableLayoutPanel3.Controls.Add(this.LoadRecordingButton, 16, 0);
            this.tableLayoutPanel3.Controls.Add(this.RecordButton, 15, 0);
            this.tableLayoutPanel3.Controls.Add(this.ToggleRendererButton, 14, 0);
            this.tableLayoutPanel3.Controls.Add(this.RadiusTextBox, 8, 0);
            this.tableLayoutPanel3.Controls.Add(this.VeloYTextBox, 7, 0);
            this.tableLayoutPanel3.Controls.Add(this.VeloXTextBox, 6, 0);
            this.tableLayoutPanel3.Controls.Add(this.AddBodiesButton, 0, 0);
            this.tableLayoutPanel3.Controls.Add(this.RemoveAllButton, 1, 0);
            this.tableLayoutPanel3.Controls.Add(this.TimeStepUpDown, 3, 0);
            this.tableLayoutPanel3.Controls.Add(this.MassTextBox, 9, 0);
            this.tableLayoutPanel3.Controls.Add(this.FlagsTextBox, 10, 0);
            this.tableLayoutPanel3.Controls.Add(this.UpdateButton, 11, 0);
            this.tableLayoutPanel3.Controls.Add(this.CenterOnMassButton, 12, 0);
            this.tableLayoutPanel3.Controls.Add(this.TrailsCheckBox, 13, 0);
            this.tableLayoutPanel3.Controls.Add(this.PauseButton, 2, 0);
            this.tableLayoutPanel3.Controls.Add(this.PressureScaleUpDown, 4, 0);
            this.tableLayoutPanel3.Controls.Add(this.AlphaUpDown, 5, 0);
            this.tableLayoutPanel3.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel3.Location = new System.Drawing.Point(3, 25);
            this.tableLayoutPanel3.Name = "tableLayoutPanel3";
            this.tableLayoutPanel3.RowCount = 1;
            this.tableLayoutPanel3.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel3.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 29F));
            this.tableLayoutPanel3.Size = new System.Drawing.Size(1577, 29);
            this.tableLayoutPanel3.TabIndex = 1;
            // 
            // LoadRecordingButton
            // 
            this.LoadRecordingButton.Dock = System.Windows.Forms.DockStyle.Left;
            this.LoadRecordingButton.Location = new System.Drawing.Point(1286, 3);
            this.LoadRecordingButton.Name = "LoadRecordingButton";
            this.LoadRecordingButton.Size = new System.Drawing.Size(83, 23);
            this.LoadRecordingButton.TabIndex = 15;
            this.LoadRecordingButton.Text = "Load Recording";
            this.LoadRecordingButton.UseVisualStyleBackColor = true;
            this.LoadRecordingButton.Click += new System.EventHandler(this.LoadRecordingButton_Click);
            // 
            // RecordButton
            // 
            this.RecordButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RecordButton.Location = new System.Drawing.Point(1176, 3);
            this.RecordButton.Name = "RecordButton";
            this.RecordButton.Size = new System.Drawing.Size(104, 23);
            this.RecordButton.TabIndex = 14;
            this.RecordButton.Text = "Record";
            this.RecordButton.UseVisualStyleBackColor = true;
            this.RecordButton.Click += new System.EventHandler(this.RecordButton_Click);
            // 
            // ToggleRendererButton
            // 
            this.ToggleRendererButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.ToggleRendererButton.Location = new System.Drawing.Point(1069, 3);
            this.ToggleRendererButton.Name = "ToggleRendererButton";
            this.ToggleRendererButton.Size = new System.Drawing.Size(101, 23);
            this.ToggleRendererButton.TabIndex = 13;
            this.ToggleRendererButton.Text = "Toggle Renderer";
            this.ToggleRendererButton.UseVisualStyleBackColor = true;
            this.ToggleRendererButton.Click += new System.EventHandler(this.ToggleRendererButton_Click);
            // 
            // RadiusTextBox
            // 
            this.RadiusTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RadiusTextBox.Location = new System.Drawing.Point(664, 3);
            this.RadiusTextBox.Name = "RadiusTextBox";
            this.RadiusTextBox.Size = new System.Drawing.Size(43, 20);
            this.RadiusTextBox.TabIndex = 7;
            // 
            // VeloYTextBox
            // 
            this.VeloYTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.VeloYTextBox.Location = new System.Drawing.Point(563, 3);
            this.VeloYTextBox.Name = "VeloYTextBox";
            this.VeloYTextBox.Size = new System.Drawing.Size(95, 20);
            this.VeloYTextBox.TabIndex = 6;
            // 
            // VeloXTextBox
            // 
            this.VeloXTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.VeloXTextBox.Location = new System.Drawing.Point(467, 3);
            this.VeloXTextBox.Name = "VeloXTextBox";
            this.VeloXTextBox.Size = new System.Drawing.Size(90, 20);
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
            this.RemoveAllButton.Click += new System.EventHandler(this.RemoveAllButton_Click);
            // 
            // TimeStepUpDown
            // 
            this.TimeStepUpDown.DecimalPlaces = 4;
            this.TimeStepUpDown.Dock = System.Windows.Forms.DockStyle.Fill;
            this.TimeStepUpDown.Increment = new decimal(new int[] {
            1,
            0,
            0,
            262144});
            this.TimeStepUpDown.Location = new System.Drawing.Point(304, 3);
            this.TimeStepUpDown.Maximum = new decimal(new int[] {
            1000,
            0,
            0,
            196608});
            this.TimeStepUpDown.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            262144});
            this.TimeStepUpDown.Name = "TimeStepUpDown";
            this.TimeStepUpDown.Size = new System.Drawing.Size(62, 20);
            this.TimeStepUpDown.TabIndex = 3;
            this.TimeStepUpDown.Value = new decimal(new int[] {
            1,
            0,
            0,
            262144});
            this.TimeStepUpDown.ValueChanged += new System.EventHandler(this.TimeStepUpDown_ValueChanged);
            // 
            // MassTextBox
            // 
            this.MassTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.MassTextBox.Location = new System.Drawing.Point(713, 3);
            this.MassTextBox.Name = "MassTextBox";
            this.MassTextBox.Size = new System.Drawing.Size(53, 20);
            this.MassTextBox.TabIndex = 8;
            // 
            // FlagsTextBox
            // 
            this.FlagsTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.FlagsTextBox.Location = new System.Drawing.Point(772, 3);
            this.FlagsTextBox.Name = "FlagsTextBox";
            this.FlagsTextBox.Size = new System.Drawing.Size(34, 20);
            this.FlagsTextBox.TabIndex = 9;
            // 
            // UpdateButton
            // 
            this.UpdateButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.UpdateButton.Location = new System.Drawing.Point(812, 3);
            this.UpdateButton.Name = "UpdateButton";
            this.UpdateButton.Size = new System.Drawing.Size(75, 23);
            this.UpdateButton.TabIndex = 10;
            this.UpdateButton.Text = "Update";
            this.UpdateButton.UseVisualStyleBackColor = true;
            this.UpdateButton.Click += new System.EventHandler(this.UpdateButton_Click);
            // 
            // CenterOnMassButton
            // 
            this.CenterOnMassButton.Dock = System.Windows.Forms.DockStyle.Fill;
            this.CenterOnMassButton.Location = new System.Drawing.Point(893, 3);
            this.CenterOnMassButton.Name = "CenterOnMassButton";
            this.CenterOnMassButton.Size = new System.Drawing.Size(89, 23);
            this.CenterOnMassButton.TabIndex = 11;
            this.CenterOnMassButton.Text = "Re-center";
            this.CenterOnMassButton.UseVisualStyleBackColor = true;
            this.CenterOnMassButton.Click += new System.EventHandler(this.CenterOnMassButton_Click);
            // 
            // TrailsCheckBox
            // 
            this.TrailsCheckBox.Appearance = System.Windows.Forms.Appearance.Button;
            this.TrailsCheckBox.AutoSize = true;
            this.TrailsCheckBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.TrailsCheckBox.Location = new System.Drawing.Point(988, 3);
            this.TrailsCheckBox.Name = "TrailsCheckBox";
            this.TrailsCheckBox.Size = new System.Drawing.Size(75, 23);
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
            this.PauseButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.PauseButton.Location = new System.Drawing.Point(207, 3);
            this.PauseButton.Name = "PauseButton";
            this.PauseButton.Size = new System.Drawing.Size(91, 23);
            this.PauseButton.TabIndex = 17;
            this.PauseButton.Text = "Pause Physics";
            this.PauseButton.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.PauseButton.UseVisualStyleBackColor = true;
            this.PauseButton.CheckedChanged += new System.EventHandler(this.PauseButton_CheckedChanged);
            // 
            // PressureScaleUpDown
            // 
            this.PressureScaleUpDown.Dock = System.Windows.Forms.DockStyle.Fill;
            this.PressureScaleUpDown.Location = new System.Drawing.Point(372, 3);
            this.PressureScaleUpDown.Maximum = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            this.PressureScaleUpDown.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.PressureScaleUpDown.Name = "PressureScaleUpDown";
            this.PressureScaleUpDown.Size = new System.Drawing.Size(42, 20);
            this.PressureScaleUpDown.TabIndex = 18;
            this.PressureScaleUpDown.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.PressureScaleUpDown.ValueChanged += new System.EventHandler(this.PressureScaleUpDown_ValueChanged);
            // 
            // AlphaUpDown
            // 
            this.AlphaUpDown.Dock = System.Windows.Forms.DockStyle.Fill;
            this.AlphaUpDown.Location = new System.Drawing.Point(420, 3);
            this.AlphaUpDown.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.AlphaUpDown.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.AlphaUpDown.Name = "AlphaUpDown";
            this.AlphaUpDown.Size = new System.Drawing.Size(41, 20);
            this.AlphaUpDown.TabIndex = 19;
            this.AlphaUpDown.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.AlphaUpDown.ValueChanged += new System.EventHandler(this.AlphaUpDown_ValueChanged);
            // 
            // panel1
            // 
            this.panel1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel1.Controls.Add(this.RendererLabel);
            this.panel1.Controls.Add(this.ScaleLabel);
            this.panel1.Controls.Add(this.RecSizeLabel);
            this.panel1.Controls.Add(this.SpeedLabel);
            this.panel1.Controls.Add(this.FrameCountLabel);
            this.panel1.Controls.Add(this.PressureLabel);
            this.panel1.Controls.Add(this.DensityLabel);
            this.panel1.Controls.Add(this.TotalMassLabel);
            this.panel1.Controls.Add(this.BodyCountLabel);
            this.panel1.Controls.Add(this.FPSLabel);
            this.panel1.Controls.Add(this.RenderBox);
            this.panel1.Location = new System.Drawing.Point(3, 66);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(1583, 746);
            this.panel1.TabIndex = 0;
            // 
            // RendererLabel
            // 
            this.RendererLabel.AutoSize = true;
            this.RendererLabel.BackColor = System.Drawing.Color.Black;
            this.RendererLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.RendererLabel.Location = new System.Drawing.Point(2, 1);
            this.RendererLabel.Name = "RendererLabel";
            this.RendererLabel.Size = new System.Drawing.Size(74, 13);
            this.RendererLabel.TabIndex = 10;
            this.RendererLabel.Text = "[RENDERER]";
            // 
            // ScaleLabel
            // 
            this.ScaleLabel.AutoSize = true;
            this.ScaleLabel.BackColor = System.Drawing.Color.Black;
            this.ScaleLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.ScaleLabel.Location = new System.Drawing.Point(3, 95);
            this.ScaleLabel.Name = "ScaleLabel";
            this.ScaleLabel.Size = new System.Drawing.Size(37, 13);
            this.ScaleLabel.TabIndex = 9;
            this.ScaleLabel.Text = "Scale:";
            // 
            // RecSizeLabel
            // 
            this.RecSizeLabel.AutoSize = true;
            this.RecSizeLabel.BackColor = System.Drawing.Color.Black;
            this.RecSizeLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.RecSizeLabel.Location = new System.Drawing.Point(3, 205);
            this.RecSizeLabel.Name = "RecSizeLabel";
            this.RecSizeLabel.Size = new System.Drawing.Size(78, 13);
            this.RecSizeLabel.TabIndex = 8;
            this.RecSizeLabel.Text = "Rec Size (MB):";
            // 
            // SpeedLabel
            // 
            this.SpeedLabel.AutoSize = true;
            this.SpeedLabel.BackColor = System.Drawing.Color.Black;
            this.SpeedLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.SpeedLabel.Location = new System.Drawing.Point(3, 147);
            this.SpeedLabel.Name = "SpeedLabel";
            this.SpeedLabel.Size = new System.Drawing.Size(44, 13);
            this.SpeedLabel.TabIndex = 7;
            this.SpeedLabel.Text = "Speed: ";
            // 
            // FrameCountLabel
            // 
            this.FrameCountLabel.AutoSize = true;
            this.FrameCountLabel.BackColor = System.Drawing.Color.Black;
            this.FrameCountLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.FrameCountLabel.Location = new System.Drawing.Point(3, 36);
            this.FrameCountLabel.Name = "FrameCountLabel";
            this.FrameCountLabel.Size = new System.Drawing.Size(38, 13);
            this.FrameCountLabel.TabIndex = 6;
            this.FrameCountLabel.Text = "Count:";
            // 
            // PressureLabel
            // 
            this.PressureLabel.AutoSize = true;
            this.PressureLabel.BackColor = System.Drawing.Color.Black;
            this.PressureLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.PressureLabel.Location = new System.Drawing.Point(3, 134);
            this.PressureLabel.Name = "PressureLabel";
            this.PressureLabel.Size = new System.Drawing.Size(36, 13);
            this.PressureLabel.TabIndex = 5;
            this.PressureLabel.Text = "Press:";
            // 
            // DensityLabel
            // 
            this.DensityLabel.AutoSize = true;
            this.DensityLabel.BackColor = System.Drawing.Color.Black;
            this.DensityLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.DensityLabel.Location = new System.Drawing.Point(3, 121);
            this.DensityLabel.Name = "DensityLabel";
            this.DensityLabel.Size = new System.Drawing.Size(45, 13);
            this.DensityLabel.TabIndex = 4;
            this.DensityLabel.Text = "Density:";
            // 
            // TotalMassLabel
            // 
            this.TotalMassLabel.AutoSize = true;
            this.TotalMassLabel.BackColor = System.Drawing.Color.Black;
            this.TotalMassLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.TotalMassLabel.Location = new System.Drawing.Point(3, 73);
            this.TotalMassLabel.Name = "TotalMassLabel";
            this.TotalMassLabel.Size = new System.Drawing.Size(54, 13);
            this.TotalMassLabel.TabIndex = 3;
            this.TotalMassLabel.Text = "Tot Mass:";
            // 
            // BodyCountLabel
            // 
            this.BodyCountLabel.AutoSize = true;
            this.BodyCountLabel.BackColor = System.Drawing.Color.Black;
            this.BodyCountLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.BodyCountLabel.Location = new System.Drawing.Point(3, 60);
            this.BodyCountLabel.Name = "BodyCountLabel";
            this.BodyCountLabel.Size = new System.Drawing.Size(42, 13);
            this.BodyCountLabel.TabIndex = 2;
            this.BodyCountLabel.Text = "Bodies:";
            // 
            // FPSLabel
            // 
            this.FPSLabel.AutoSize = true;
            this.FPSLabel.BackColor = System.Drawing.Color.Black;
            this.FPSLabel.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(192)))), ((int)(((byte)(0)))));
            this.FPSLabel.Location = new System.Drawing.Point(3, 23);
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
            this.RenderBox.Size = new System.Drawing.Size(1581, 744);
            this.RenderBox.TabIndex = 0;
            this.RenderBox.TabStop = false;
            this.RenderBox.MouseDown += new System.Windows.Forms.MouseEventHandler(this.RenderBox_MouseDown);
            this.RenderBox.MouseMove += new System.Windows.Forms.MouseEventHandler(this.RenderBox_MouseMove);
            this.RenderBox.MouseUp += new System.Windows.Forms.MouseEventHandler(this.RenderBox_MouseUp);
            this.RenderBox.Resize += new System.EventHandler(this.RenderBox_Resize);
            // 
            // DisplayForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1589, 815);
            this.Controls.Add(this.RootLayoutTable);
            this.DoubleBuffered = true;
            this.KeyPreview = true;
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "DisplayForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "NBodies";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.DisplayForm_FormClosing);
            this.Load += new System.EventHandler(this.DisplayForm_Load);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.DisplayForm_KeyDown);
            this.KeyUp += new System.Windows.Forms.KeyEventHandler(this.DisplayForm_KeyUp);
            this.RootLayoutTable.ResumeLayout(false);
            this.tableLayoutPanel2.ResumeLayout(false);
            this.tableLayoutPanel2.PerformLayout();
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.tableLayoutPanel3.ResumeLayout(false);
            this.tableLayoutPanel3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.TimeStepUpDown)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.PressureScaleUpDown)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.AlphaUpDown)).EndInit();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.RenderBox)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TableLayoutPanel RootLayoutTable;
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
        private System.Windows.Forms.TextBox MassTextBox;
        private System.Windows.Forms.TextBox FlagsTextBox;
        private System.Windows.Forms.Button UpdateButton;
        private System.Windows.Forms.Button CenterOnMassButton;
        private System.Windows.Forms.Button LoadRecordingButton;
        private System.Windows.Forms.Button RecordButton;
        private System.Windows.Forms.Button ToggleRendererButton;
        private System.Windows.Forms.CheckBox TrailsCheckBox;
        private System.Windows.Forms.CheckBox PauseButton;
        private System.Windows.Forms.ToolStripMenuItem saveStateToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem loadStateToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem antiAliasingToolStripMenuItem;
        private System.Windows.Forms.Label FPSLabel;
        private System.Windows.Forms.Label BodyCountLabel;
        private System.Windows.Forms.Label TotalMassLabel;
        private System.Windows.Forms.Label PressureLabel;
        private System.Windows.Forms.Label DensityLabel;
        private System.Windows.Forms.ToolStripMenuItem displayToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem normalToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem pressuresToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem highContrastToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem clipToViewportToolStripMenuItem;
        private System.Windows.Forms.NumericUpDown PressureScaleUpDown;
        private System.Windows.Forms.Label SpeedLabel;
        private System.Windows.Forms.ToolStripMenuItem reloadPreviousToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem speedsToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem forcesToolStripMenuItem;
        private System.Windows.Forms.NumericUpDown AlphaUpDown;
        private System.Windows.Forms.ToolStripMenuItem followingBodyToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem showFollowBodyForce;
        private System.Windows.Forms.ToolStripMenuItem showPredictOrbit;
        private System.Windows.Forms.ToolStripMenuItem drawToolStripMenuItem;
        private System.Windows.Forms.Label RecSizeLabel;
        private System.Windows.Forms.ToolStripMenuItem rocheLimitToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem showMeshToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem allForceVectorsToolStripMenuItem;
        private System.Windows.Forms.Label ScaleLabel;
        private System.Windows.Forms.Label RendererLabel;
        private System.Windows.Forms.Label FrameCountLabel;
        private System.Windows.Forms.ToolStripMenuItem indexToolStripMenuItem;
    }
}

