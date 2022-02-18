typedef struct
{
	float PosX;
	float PosY;
	float PosZ;
	float Mass;
	float VeloX;
	float VeloY;
	float VeloZ;
	float ForceX;
	float ForceY;
	float ForceZ;
	int Color;
	float Size;
	int Flag;
	int UID;
	float Density;
	float Temp;
	float Lifetime;
	int MeshID;

} Body;

typedef struct
{
	float kSize;
	float kSizeSq;
	float kSize3;
	float kRad6;
	float kSize9;
	float fViscosity;
	float fPressure;
	float fDensity;

} SPHPreCalc;

typedef struct
{
	float KernelSize;
	float DeltaTime;
	float Viscosity;
	float CullDistance;
	float GasK;
	int CollisionsOn;
	int MeshLevels;
	int CellSizeExponent;

} SimSettings;

// Grav/SPH consts.
constant float SPH_SOFTENING = 0.00001f;
constant float SOFTENING = 0.04f;
constant float SOFTENING_SQRT = 0.2f;

// Flags
constant int BLACKHOLE = 1;
constant int ISEXPLOSION = 2;
constant int CULLED = 4;
constant int INROCHE = 8;


#if FASTMATH

#define DISTANCE(a,b) fast_distance(a,b)
#define SQRT(a) half_sqrt(a)
#define LENGTH(a) fast_length(a)

#else

#define DISTANCE(a,b) distance(a,b)
#define SQRT(a) sqrt(a)
#define LENGTH(a) length(a)

#endif

// Sorting defs.
#define PADDING_ELEM -1

long MortonNumber(long x, long y, long z);
float3 ComputeForce(float3 posA, float3 posB, float massA, float massB);
bool IsFar(int4 cell, int4 testCell);
Body CollideBodies(Body master, Body slave, float colMass, float forceX, float forceY, float forceZ);
int SetFlag(int flags, int flag, bool enabled);
Body SetFlagB(Body body, int flag, bool enabled);
bool HasFlag(int flags, int check);
bool HasFlagB(Body body, int check);
int BlockCount(int len, int threads);