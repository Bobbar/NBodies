using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies
{
    public static class Logger
    {
        public static bool Enabled = false;

        public static void Out(string message)
        {
            if (Enabled)
                Console.WriteLine(message);
        }
    }
}
