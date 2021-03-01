using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public struct SPHPreCalc
    {
        public float kSize;
        public float kSizeSq;
        public float kSize3;
        public float kRad6;
        public float kSize9;
        public float fViscosity;
        public float fPressure;
        public float fDensity;

        public string ToString()
        {
            return $"kSize: {kSize}  kSizeSq: {kSizeSq}  kSize3: {kSize3}  kRad6: {kRad6}  kSize9: {kSize9}  Visc: {fViscosity}  Press: {fPressure}  Dens: {fDensity}";
        }
    }
}
