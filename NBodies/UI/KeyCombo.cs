using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NBodies.UI
{
    public class KeyCombo : IEquatable<KeyCombo>
    {
        public readonly List<Keys> Keys = new List<Keys>();

        public KeyCombo()
        { }

        public KeyCombo(Keys key)
        {
            Keys.Add(key);
        }

        //public KeyCombo(Keys[] keys)
        //{
        //    Keys.AddRange(keys);
        //}

        public KeyCombo(params Keys[] keys)
        {
            Keys.AddRange(keys);
        }

        public KeyCombo(List<Keys> keys)
        {
            Keys.AddRange(keys);
        }

        public void AddKey(Keys key)
        {
            Keys.Add(key);
        }

        public bool Equals(KeyCombo other)
        {
            if (Keys.Count != other.Keys.Count)
                return false;


            else
            {
                foreach (var key in other.Keys)
                {
                    if (!Keys.Contains(key))
                        return false;
                }


                //for (int i = 0; i < Keys.Count; i++)
                //{
                //    if (Keys[i] != other.Keys[i])
                //        return false;
                //}
            }



            return true;
        }

        public bool Contains(Keys key)
        {
            foreach (var k in Keys)
                if (k == key)
                    return true;

            return false;
        }
    }
}
