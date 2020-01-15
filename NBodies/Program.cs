using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using NBodies.UI;
using NBodies.Rendering;

namespace NBodies
{
    static class Program
    {
        public static int ThreadsPerBlockArgument = -1;
        public static int DeviceID = -1;
      
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            var args = Environment.GetCommandLineArgs();

            bool tpbSet = false;
            bool devIdSet = false;

            foreach (var arg in args)
            {

                if (!tpbSet)
                {
                    int tpb = 1;

                    if (int.TryParse(arg, out tpb))
                    {
                        ThreadsPerBlockArgument = tpb;
                        tpbSet = true;
                        continue;
                    }
                }

                if (tpbSet && !devIdSet)
                {
                    int devid = 0;

                    if (int.TryParse(arg, out devid))
                    {
                        DeviceID = devid;
                        devIdSet = true;
                        continue;
                    }
                }
                
                
            }

             Application.Run(new DisplayForm());


        }
    }
}
