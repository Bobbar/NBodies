using System;
using System.Reflection;
using System.Windows.Forms;
using System.Diagnostics;

namespace NBodies.Extensions
{
    public static class ControlExtentions
    {
        /// <summary>
        /// Sets the protected Control.DoubleBuffered property. Does not set if we are running within a terminal session (RDP).
        /// </summary>
        /// <param name="control"></param>
        /// <param name="setting"></param>
        public static void DoubleBuffered(this Control control, bool setting)
        {
            if (SystemInformation.TerminalServerSession) return;
            Type type = control.GetType();
            PropertyInfo pi = type.GetProperty("DoubleBuffered", BindingFlags.Instance | BindingFlags.NonPublic);
            pi.SetValue(control, setting, null);
        }

        public static void Print(this Stopwatch timer, string label = "")
        {
            Debug.WriteLine($@"[{label}] ms: {timer.ElapsedMilliseconds}  ticks: {timer.Elapsed.Ticks}");

            timer.Restart();
        }

        public static void Log(this Stopwatch timer, string label = "")
        {
            System.IO.File.AppendAllText($@".\TimerLog.txt", $@"[{label}] ms: {timer.ElapsedMilliseconds}  ticks: {timer.Elapsed.Ticks} {Environment.NewLine}");

            timer.Restart();
        }

    }
}
