typedef struct
{
	float PosX;
	float PosY;
	float Mass;
	float VeloX;
	float VeloY;
	float ForceX;
	float ForceY;
	int Color;
	float Size;
	int Flag;
	int UID;
	float Density;
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
constant float SOFTENING = 0.004f;

// Flags
constant int BLACKHOLE = 1;
constant int ISEXPLOSION = 2;
constant int CULLED = 4;
constant int INROCHE = 8;

// Lookup table for neighbor search.
constant int2 N_OFFSET_LUT[] = { { -1,-1 }, { 0,-1 }, { 1,-1 }, { -1,0 }, { 1,0 }, { -1,1 }, { 0,1 }, { 1,1 } };

#define DISTANCE(a,b) distance(a,b)
#define SQRT(a) sqrt(a)
#define LENGTH(a) length(a)

// Padding element for sorting.
#define PADDING_ELEM -1

long MortonNumberInt2(int2 idx);
int BinarySearch(global int2* meshIdxs, int2 cellIdx, int start, int end);
float2 CellForce(float2 posA, float2 posB, float massA, float massB);
bool IsFar(int2 cell, int2 testCell);
Body CollideBodies(Body master, Body slave, float colMass, float forceX, float forceY);
int SetFlag(int flags, int flag, bool enabled);
Body SetFlagB(Body body, int flag, bool enabled);
bool HasFlag(int flags, int check);
bool HasFlagB(Body body, int check);
int BlockCount(int len, int threads);
