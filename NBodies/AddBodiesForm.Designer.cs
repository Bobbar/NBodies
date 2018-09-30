namespace NBodies
{
    partial class AddBodiesForm
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
            this.BodyParamsGroup = new System.Windows.Forms.GroupBox();
            this.StationaryGroup = new System.Windows.Forms.GroupBox();
            this.AddStationaryButton = new System.Windows.Forms.Button();
            this.CirOrbitGroup = new System.Windows.Forms.GroupBox();
            this.AddOrbitButton = new System.Windows.Forms.Button();
            this.label8 = new System.Windows.Forms.Label();
            this.OrbitRadiusTextBox = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.CenterMassTextBox = new System.Windows.Forms.TextBox();
            this.CenterMassCheckBox = new System.Windows.Forms.CheckBox();
            this.panel1 = new System.Windows.Forms.Panel();
            this.label6 = new System.Windows.Forms.Label();
            this.DensityTextBox = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.MassTextBox = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.MaxSizeTextBox = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.MinSizeTextBox = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.NumToAddTextBox = new System.Windows.Forms.TextBox();
            this.BodyParamsGroup.SuspendLayout();
            this.StationaryGroup.SuspendLayout();
            this.CirOrbitGroup.SuspendLayout();
            this.panel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // BodyParamsGroup
            // 
            this.BodyParamsGroup.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.BodyParamsGroup.Controls.Add(this.StationaryGroup);
            this.BodyParamsGroup.Controls.Add(this.CirOrbitGroup);
            this.BodyParamsGroup.Controls.Add(this.panel1);
            this.BodyParamsGroup.Location = new System.Drawing.Point(12, 12);
            this.BodyParamsGroup.Name = "BodyParamsGroup";
            this.BodyParamsGroup.Size = new System.Drawing.Size(386, 356);
            this.BodyParamsGroup.TabIndex = 0;
            this.BodyParamsGroup.TabStop = false;
            this.BodyParamsGroup.Text = "Bodies";
            // 
            // StationaryGroup
            // 
            this.StationaryGroup.Controls.Add(this.AddStationaryButton);
            this.StationaryGroup.Location = new System.Drawing.Point(6, 263);
            this.StationaryGroup.Name = "StationaryGroup";
            this.StationaryGroup.Size = new System.Drawing.Size(374, 85);
            this.StationaryGroup.TabIndex = 2;
            this.StationaryGroup.TabStop = false;
            this.StationaryGroup.Text = "Add Stationary";
            // 
            // AddStationaryButton
            // 
            this.AddStationaryButton.Location = new System.Drawing.Point(109, 19);
            this.AddStationaryButton.Name = "AddStationaryButton";
            this.AddStationaryButton.Size = new System.Drawing.Size(145, 47);
            this.AddStationaryButton.TabIndex = 9;
            this.AddStationaryButton.Text = "Add Stationary";
            this.AddStationaryButton.UseVisualStyleBackColor = true;
            this.AddStationaryButton.Click += new System.EventHandler(this.AddStationaryButton_Click);
            // 
            // CirOrbitGroup
            // 
            this.CirOrbitGroup.Controls.Add(this.AddOrbitButton);
            this.CirOrbitGroup.Controls.Add(this.label8);
            this.CirOrbitGroup.Controls.Add(this.OrbitRadiusTextBox);
            this.CirOrbitGroup.Controls.Add(this.label7);
            this.CirOrbitGroup.Controls.Add(this.CenterMassTextBox);
            this.CirOrbitGroup.Controls.Add(this.CenterMassCheckBox);
            this.CirOrbitGroup.Location = new System.Drawing.Point(6, 103);
            this.CirOrbitGroup.Name = "CirOrbitGroup";
            this.CirOrbitGroup.Size = new System.Drawing.Size(374, 151);
            this.CirOrbitGroup.TabIndex = 1;
            this.CirOrbitGroup.TabStop = false;
            this.CirOrbitGroup.Text = "Add Into Circular Orbit";
            // 
            // AddOrbitButton
            // 
            this.AddOrbitButton.Location = new System.Drawing.Point(171, 59);
            this.AddOrbitButton.Name = "AddOrbitButton";
            this.AddOrbitButton.Size = new System.Drawing.Size(145, 47);
            this.AddOrbitButton.TabIndex = 8;
            this.AddOrbitButton.Text = "Add Orbit";
            this.AddOrbitButton.UseVisualStyleBackColor = true;
            this.AddOrbitButton.Click += new System.EventHandler(this.AddOrbitButton_Click);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(31, 93);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(40, 13);
            this.label8.TabIndex = 7;
            this.label8.Text = "Radius";
            // 
            // OrbitRadiusTextBox
            // 
            this.OrbitRadiusTextBox.Location = new System.Drawing.Point(34, 109);
            this.OrbitRadiusTextBox.Name = "OrbitRadiusTextBox";
            this.OrbitRadiusTextBox.Size = new System.Drawing.Size(63, 20);
            this.OrbitRadiusTextBox.TabIndex = 6;
            this.OrbitRadiusTextBox.Text = "1000";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(31, 49);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(66, 13);
            this.label7.TabIndex = 5;
            this.label7.Text = "Center Mass";
            // 
            // CenterMassTextBox
            // 
            this.CenterMassTextBox.Location = new System.Drawing.Point(34, 65);
            this.CenterMassTextBox.Name = "CenterMassTextBox";
            this.CenterMassTextBox.Size = new System.Drawing.Size(63, 20);
            this.CenterMassTextBox.TabIndex = 4;
            this.CenterMassTextBox.Text = "8000";
            // 
            // CenterMassCheckBox
            // 
            this.CenterMassCheckBox.AutoSize = true;
            this.CenterMassCheckBox.Checked = true;
            this.CenterMassCheckBox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.CenterMassCheckBox.Location = new System.Drawing.Point(15, 29);
            this.CenterMassCheckBox.Name = "CenterMassCheckBox";
            this.CenterMassCheckBox.Size = new System.Drawing.Size(123, 17);
            this.CenterMassCheckBox.TabIndex = 0;
            this.CenterMassCheckBox.Text = "Include Center Mass";
            this.CenterMassCheckBox.UseVisualStyleBackColor = true;
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.label6);
            this.panel1.Controls.Add(this.DensityTextBox);
            this.panel1.Controls.Add(this.label5);
            this.panel1.Controls.Add(this.label4);
            this.panel1.Controls.Add(this.MassTextBox);
            this.panel1.Controls.Add(this.label3);
            this.panel1.Controls.Add(this.MaxSizeTextBox);
            this.panel1.Controls.Add(this.label2);
            this.panel1.Controls.Add(this.MinSizeTextBox);
            this.panel1.Controls.Add(this.label1);
            this.panel1.Controls.Add(this.NumToAddTextBox);
            this.panel1.Location = new System.Drawing.Point(6, 16);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(361, 81);
            this.panel1.TabIndex = 0;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(274, 13);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(42, 13);
            this.label6.TabIndex = 10;
            this.label6.Text = "Density";
            // 
            // DensityTextBox
            // 
            this.DensityTextBox.Location = new System.Drawing.Point(277, 29);
            this.DensityTextBox.Name = "DensityTextBox";
            this.DensityTextBox.Size = new System.Drawing.Size(70, 20);
            this.DensityTextBox.TabIndex = 9;
            this.DensityTextBox.Text = "0";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label5.Location = new System.Drawing.Point(207, 52);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(47, 13);
            this.label5.TabIndex = 8;
            this.label5.Text = "0 = Auto";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(198, 13);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(32, 13);
            this.label4.TabIndex = 7;
            this.label4.Text = "Mass";
            // 
            // MassTextBox
            // 
            this.MassTextBox.Location = new System.Drawing.Point(201, 29);
            this.MassTextBox.Name = "MassTextBox";
            this.MassTextBox.Size = new System.Drawing.Size(70, 20);
            this.MassTextBox.TabIndex = 6;
            this.MassTextBox.Text = "0";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(145, 13);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(50, 13);
            this.label3.TabIndex = 5;
            this.label3.Text = "Max Size";
            // 
            // MaxSizeTextBox
            // 
            this.MaxSizeTextBox.Location = new System.Drawing.Point(148, 29);
            this.MaxSizeTextBox.Name = "MaxSizeTextBox";
            this.MaxSizeTextBox.Size = new System.Drawing.Size(47, 20);
            this.MaxSizeTextBox.TabIndex = 4;
            this.MaxSizeTextBox.Text = "5";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(88, 13);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(47, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "Min Size";
            // 
            // MinSizeTextBox
            // 
            this.MinSizeTextBox.Location = new System.Drawing.Point(91, 29);
            this.MinSizeTextBox.Name = "MinSizeTextBox";
            this.MinSizeTextBox.Size = new System.Drawing.Size(51, 20);
            this.MinSizeTextBox.TabIndex = 2;
            this.MinSizeTextBox.Text = "1";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 13);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(48, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "# to Add";
            // 
            // NumToAddTextBox
            // 
            this.NumToAddTextBox.Location = new System.Drawing.Point(15, 29);
            this.NumToAddTextBox.Name = "NumToAddTextBox";
            this.NumToAddTextBox.Size = new System.Drawing.Size(70, 20);
            this.NumToAddTextBox.TabIndex = 0;
            this.NumToAddTextBox.Text = "500";
            // 
            // AddBodiesForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(410, 380);
            this.Controls.Add(this.BodyParamsGroup);
            this.Name = "AddBodiesForm";
            this.Text = "Add Bodies";
            this.BodyParamsGroup.ResumeLayout(false);
            this.StationaryGroup.ResumeLayout(false);
            this.CirOrbitGroup.ResumeLayout(false);
            this.CirOrbitGroup.PerformLayout();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox BodyParamsGroup;
        private System.Windows.Forms.GroupBox StationaryGroup;
        private System.Windows.Forms.Button AddStationaryButton;
        private System.Windows.Forms.GroupBox CirOrbitGroup;
        private System.Windows.Forms.Button AddOrbitButton;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox OrbitRadiusTextBox;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox CenterMassTextBox;
        private System.Windows.Forms.CheckBox CenterMassCheckBox;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox DensityTextBox;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox MassTextBox;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox MaxSizeTextBox;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox MinSizeTextBox;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox NumToAddTextBox;
    }
}