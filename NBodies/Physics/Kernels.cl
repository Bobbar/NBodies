

typedef struct __attribute__((packed)) Body
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
	float Pressure;
	float Lifetime;
	int MeshID;

} Body;

typedef struct MeshCell
{
	int ID;
	int IdxX;
	int IdxY;
	int IdxZ;
	float CmX;
	float CmY;
	float CmZ;
	float Mass;
	int Size;
	int BodyStartIdx;
	int BodyCount;
	int NeighborStartIdx;
	int NeighborCount;
	int ChildStartIdx;
	int ChildCount;
	int ParentID;
	int Level;
	long GridIdx;

} MeshCell;

typedef struct GridInfo
{
	int OffsetX;
	int OffsetY;
	int OffsetZ;
	int MinX;
	int MinY;
	int MinZ;
	int MaxX;
	int MaxY;
	int MaxZ;
	long Columns;
	long Rows;
	long Layers;
	long Size;
	long IndexOffset;

} GridInfo;

typedef struct __attribute__((packed)) SPHPreCalc
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

typedef struct __attribute__((packed)) SimSettings
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
#define SQRT(a) native_sqrt(a)

#else

#define DISTANCE(a,b) distance(a,b)
#define SQRT(a) sqrt(a)

#endif


float3 ComputeForce(float3 posA, float3 posB, float massA, float massB);
long GridHash(long x, long y, long z, GridInfo grid);
bool IsFar(MeshCell cell, MeshCell testCell);
Body CollideBodies(Body master, Body slave, float colMass, float forceX, float forceY, float forceZ);
int SetFlag(int flags, int flag, bool enabled);
Body SetFlagB(Body body, int flag, bool enabled);
bool HasFlag(int flags, int check);
bool HasFlagB(Body body, int check);


int SetFlag(int flags, int flag, bool enabled)
{
	if (HasFlag(flags, flag))
	{
		if (!enabled)
			return flags -= flag;
	}
	else
	{
		if (enabled)
			return flags += flag;
	}

	return flags;
}

Body SetFlagB(Body body, int flag, bool enabled)
{
	Body out = body;

	if (HasFlag(out.Flag, flag))
	{
		if (!enabled)
			out.Flag -= flag;
	}
	else
	{
		if (enabled)
			out.Flag += flag;
	}

	return out;
}

bool HasFlag(int flags, int check)
{
	return (check & flags) != 0;
}

bool HasFlagB(Body body, int check)
{
	return (check & body.Flag) != 0;
}


__kernel void FixOverlaps(global Body* inBodies, int inBodiesLen, global Body* outBodies)
{
	int i = get_global_id(0);

	if (i >= inBodiesLen)
	{
		return;
	}

	Body bodyA = inBodies[i];

	for (int j = 0; j < inBodiesLen; j++)
	{
		if (i != j)
		{
			Body bodyB = inBodies[j];
			float aRad = bodyA.Size * 0.5f;
			float bRad = bodyB.Size * 0.5f;
			float distX = bodyA.PosX - bodyB.PosX;
			float distY = bodyA.PosY - bodyB.PosY;
			float distZ = bodyA.PosZ - bodyB.PosZ;

			float dist = (distX * distX) + (distY * distY) + (distZ * distZ);
			float colDist = aRad + bRad;

			if (dist <= (colDist * colDist))
			{
				float distSqrt = (float)native_sqrt(dist);
				float overlap = 0.5f * (distSqrt - aRad - bRad);

				bodyA.PosX -= overlap * (distX) / distSqrt;
				bodyA.PosY -= overlap * (distY) / distSqrt;
				bodyA.PosZ -= overlap * (distZ) / distSqrt;

			}
		}
	}

	outBodies[i] = bodyA;
}


__kernel void ReindexBodies(global Body* inBodies, int blen, global int* sortMap, global Body* outBodies)
{
	int b = get_global_id(0);

	if (b >= blen)
		return;

	outBodies[b] = inBodies[sortMap[b]];
}


__kernel void ClearGrid(global int* gridIdx, long passStride, long passOffset, global MeshCell* mesh, int meshLen, int topStart)
{
	int m = get_global_id(0) + topStart;

	if (m >= meshLen)
	{
		return;
	}

	int idx = mesh[m].GridIdx;
	idx -= passOffset;

	if (idx >= 0 && idx < passStride)
	{
		gridIdx[idx] = -1;
	}
}

long GridHash(long x, long y, long z, GridInfo grid)
{
	long column = x + grid.OffsetX;
	long row = y + grid.OffsetY;
	long layer = z + grid.OffsetZ;

	return ((layer * grid.Rows) * grid.Columns) + (row * grid.Columns) + column;
}

__kernel void PopGrid(global int* gridIdx, long passStride, long passOffset, global GridInfo* gridInfo, global MeshCell* mesh, int meshLen, int topStart)
{
	int m = get_global_id(0);
	int readM = m + topStart;

	if (readM >= meshLen)
		return;

	MeshCell cell;
	cell.IdxX = mesh[readM].IdxX;
	cell.IdxY = mesh[readM].IdxY;
	cell.IdxZ = mesh[readM].IdxZ;
	cell.Level = mesh[readM].Level;
	cell.GridIdx = 0;

	GridInfo grid;
	grid.OffsetX = gridInfo[cell.Level].OffsetX;
	grid.OffsetY = gridInfo[cell.Level].OffsetY;
	grid.OffsetZ = gridInfo[cell.Level].OffsetZ;
	grid.Rows = gridInfo[cell.Level].Rows;
	grid.Columns = gridInfo[cell.Level].Columns;
	grid.IndexOffset = gridInfo[cell.Level].IndexOffset;

	// Compute bucket index.
	long bucket = GridHash(cell.IdxX, cell.IdxY, cell.IdxZ, grid);

	// Offset bucket for this level.
	bucket += grid.IndexOffset;

	// Set cell grid index to the actual bucket location.
	cell.GridIdx = bucket;

	// Offset bucket index with the pass offset. 
	// This offset is greater than 0 only when more than 1 pass is needed.
	bucket -= passOffset;

	// Make sure the bucket fits withing this pass.
	if (bucket >= 0 && bucket < passStride)
	{
		gridIdx[bucket] = readM;
	}

	mesh[readM].GridIdx = cell.GridIdx;
}


__kernel void BuildNeighbors(global MeshCell* mesh, int meshLen, global GridInfo* gridInfo, global int* gridIdx, long passStride, long passOffset, global int* neighborIndex, int topStart)
{
	int m = get_global_id(0);
	int readM = m + topStart;

	if (readM >= meshLen)
		return;

	long offset = m * 27;

	int cellLevel = mesh[readM].Level;
	long count = mesh[readM].NeighborCount;
	int3 Idx = (int3)(mesh[readM].IdxX, mesh[readM].IdxY, mesh[readM].IdxZ);

	GridInfo grid = gridInfo[cellLevel];

	int3 shiftLut[] = { { -1,-1,-1 },{ -1,-1,0 },{ -1,-1,1 },{ -1,0,-1 },{ -1,0,0 },{ -1,0,1 },{ -1,1,-1 },{ -1,1,0 },{ -1,1,1 },{ 0,-1,-1 },{ 0,-1,0 },{ 0,-1,1 },{ 0,0,-1 },{ 0,0,0 },{ 0,0,1 },{ 0,1,-1 },{ 0,1,0 },{ 0,1,1 },{ 1,-1,-1 },{ 1,-1,0 },{ 1,-1,1 },{ 1,0,-1 },{ 1,0,0 },{ 1,0,1 },{ 1,1,-1 },{ 1,1,0 },{ 1,1,1 } };

	// Shift bucket index around the cell and check for populated grid index buckets.
	for (int s = 0; s < 27; s++)
	{
		int3 sIdx = Idx + shiftLut[s];
		long localIdx = GridHash(sIdx.x, sIdx.y, sIdx.z, grid);
		if (localIdx > 0 && localIdx < grid.Size)
		{
			long bucket = localIdx + grid.IndexOffset - passOffset;
			if (bucket >= 0 && bucket < passStride)
			{
				long idx = gridIdx[bucket];
				// Check for populated bucket and poplate neighbor index.
				if (idx >= 0)
				{
					neighborIndex[(offset + count++)] = idx;
				}
			}
		}
	}

	mesh[readM].NeighborStartIdx = offset;
	mesh[readM].NeighborCount = count;
}

__kernel void BuildNeighborsBrute(global MeshCell* mesh, int meshLen, global int* levelIdx, global int* neighborIndex, int levels, int topStart)
{
	int m = get_global_id(0);
	int readM = m + topStart;

	if (readM >= meshLen)
		return;

	long offset = m * 27;

	MeshCell cell;
	cell.IdxX = mesh[readM].IdxX;
	cell.IdxY = mesh[readM].IdxY;
	cell.IdxZ = mesh[readM].IdxZ;
	cell.Level = mesh[readM].Level;

	long count = 0;
	int levelStart = levelIdx[levels];
	int levelEnd = meshLen;

	if (cell.Level < levels)
	{
		levelStart = levelIdx[cell.Level];
		levelEnd = levelIdx[cell.Level + 1];
	}

	for (int i = levelStart; i < levelEnd; i++)
	{
		MeshCell check;
		check.IdxX = mesh[i].IdxX;
		check.IdxY = mesh[i].IdxY;
		check.IdxZ = mesh[i].IdxZ;

		if (!IsFar(cell, check))
		{
			neighborIndex[(offset + count++)] = i;
		}
	}

	mesh[readM].NeighborStartIdx = offset;
	mesh[readM].NeighborCount = count;
}

// Top-down nearest neighbor search.
__kernel void BuildNeighborsBruteTD(global MeshCell* mesh, global int* levelIdx, global int* neighborIndex, int levels, int level, int start, int end)
{
	int m = get_global_id(0);
	int readM = m + start;

	if (readM >= end)
		return;

	long offset = (readM - levelIdx[1]) * 27;
	long count = 0;

	MeshCell cell = mesh[readM];

	if (level < levels)
	{
		// Use the mesh tree hierarchy & neighbors found for parent level to narrow down the search area significantly.
		MeshCell cellParent = mesh[cell.ParentID];

		// Iterate parent cell neighbors.
		int start = cellParent.NeighborStartIdx;
		int len = start + cellParent.NeighborCount;
		for (int nc = start; nc < len; nc++)
		{
			// Iterate neighbor child cells.
			int nId = neighborIndex[(nc)];
			int childStartIdx = mesh[(nId)].ChildStartIdx;
			int childLen = childStartIdx + mesh[(nId)].ChildCount;
			for (int c = childStartIdx; c < childLen; c++)
			{
				MeshCell child = mesh[(c)];

				// Check for neighbors and add them to the list.
				if (!IsFar(cell, child))
				{
					neighborIndex[(offset + count++)] = c;
				}
			}
		}
	}
	else
	{
		// For the top-most level, iterate all cells. (There won't be many, so this is fast.)
		for (int i = start; i < end; i++)
		{
			MeshCell check;
			check.IdxX = mesh[i].IdxX;
			check.IdxY = mesh[i].IdxY;
			check.IdxZ = mesh[i].IdxZ;

			if (!IsFar(cell, check))
			{
				neighborIndex[(offset + count++)] = i;
			}
		}
	}

	mesh[readM].NeighborStartIdx = offset;
	mesh[readM].NeighborCount = count;
}


__kernel void BuildBottom(global Body* inBodies, global MeshCell* mesh, int meshLen, global int* cellIdx, int cellSizeExp, int cellSize)
{
	int m = get_global_id(0);

	if (m >= meshLen)
		return;

	int firstIdx = cellIdx[m];
	int lastIdx = cellIdx[m + 1];

	float fPosX = inBodies[firstIdx].PosX;
	float fPosY = inBodies[firstIdx].PosY;
	float fPosZ = inBodies[firstIdx].PosZ;
	float fMass = inBodies[firstIdx].Mass;

	MeshCell newCell;
	newCell.IdxX = (int)floor(fPosX) >> cellSizeExp;
	newCell.IdxY = (int)floor(fPosY) >> cellSizeExp;
	newCell.IdxZ = (int)floor(fPosZ) >> cellSizeExp;
	newCell.Size = cellSize;
	newCell.BodyStartIdx = firstIdx;
	newCell.BodyCount = 1;
	newCell.ChildCount = 0;
	newCell.ID = m;
	newCell.Level = 0;
	newCell.ParentID = -1;
	newCell.CmX = fMass * fPosX;
	newCell.CmY = fMass * fPosY;
	newCell.CmZ = fMass * fPosZ;
	newCell.Mass = fMass;
	newCell.NeighborStartIdx = -1;
	newCell.NeighborCount = 0;

	inBodies[firstIdx].MeshID = m;

	for (int i = firstIdx + 1; i < lastIdx; i++)
	{
		float posX = inBodies[i].PosX;
		float posY = inBodies[i].PosY;
		float posZ = inBodies[i].PosZ;
		float mass = inBodies[i].Mass;

		newCell.Mass += mass;
		newCell.CmX += mass * posX;
		newCell.CmY += mass * posY;
		newCell.CmZ += mass * posZ;

		newCell.BodyCount++;

		inBodies[i].MeshID = m;
	}

	newCell.CmX = newCell.CmX / newCell.Mass;
	newCell.CmY = newCell.CmY / newCell.Mass;
	newCell.CmZ = newCell.CmZ / newCell.Mass;

	mesh[m] = newCell;
}


__kernel void BuildTop(global MeshCell* mesh, int len, global int* cellIdx, int cellSizeExp, int cellSize, int levelOffset, int cellIdxOffset, int level)
{
	int m = get_global_id(0);

	if (m >= len)
		return;

	int cellIdxOff = m + cellIdxOffset;
	int newIdx = m + (cellIdxOffset - level);

	int firstIdx = cellIdx[cellIdxOff] + levelOffset;
	int lastIdx = cellIdx[cellIdxOff + 1] + levelOffset;
	float mass = mesh[firstIdx].Mass;

	MeshCell newCell;
	newCell.IdxX = mesh[firstIdx].IdxX >> 1;
	newCell.IdxY = mesh[firstIdx].IdxY >> 1;
	newCell.IdxZ = mesh[firstIdx].IdxZ >> 1;
	newCell.Size = cellSize;
	newCell.ChildStartIdx = firstIdx;
	newCell.ChildCount = 1;
	newCell.ID = newIdx;
	newCell.Level = level;
	newCell.BodyStartIdx = mesh[firstIdx].BodyStartIdx;
	newCell.BodyCount = mesh[firstIdx].BodyCount;
	newCell.ParentID = -1;
	newCell.CmX = mass * mesh[firstIdx].CmX;
	newCell.CmY = mass * mesh[firstIdx].CmY;
	newCell.CmZ = mass * mesh[firstIdx].CmZ;
	newCell.Mass = mass;
	newCell.NeighborStartIdx = -1;
	newCell.NeighborCount = 0;

	mesh[firstIdx].ParentID = newIdx;

	for (int i = firstIdx + 1; i < lastIdx; i++)
	{
		float mass = mesh[i].Mass;

		newCell.Mass += mass;
		newCell.CmX += mass * mesh[i].CmX;
		newCell.CmY += mass * mesh[i].CmY;
		newCell.CmZ += mass * mesh[i].CmZ;

		newCell.ChildCount++;
		newCell.BodyCount += mesh[i].BodyCount;

		mesh[i].ParentID = newIdx;
	}

	newCell.CmX = newCell.CmX / newCell.Mass;
	newCell.CmY = newCell.CmY / newCell.Mass;
	newCell.CmZ = newCell.CmZ / newCell.Mass;


	mesh[newIdx] = newCell;
}


__kernel void CalcCenterOfMass(global MeshCell* inMesh, global float3* cm, int start, int end)
{
	double cmX = 0;
	double cmY = 0;
	double cmZ = 0;

	double mass = 0;

	for (int i = start; i < end; i++)
	{
		MeshCell cell = inMesh[i];

		mass += cell.Mass;
		cmX += cell.Mass * cell.CmX;
		cmY += cell.Mass * cell.CmY;
		cmZ += cell.Mass * cell.CmZ;

	}

	cmX = cmX / mass;
	cmY = cmY / mass;
	cmZ = cmZ / mass;


	cm[0] = (float3)(cmX, cmY, cmZ);
}


float3 ComputeForce(float3 posA, float3 posB, float massA, float massB)
{
	float3 dir = posA - posB;
	float dist = dot(dir, dir);
	float distSqrt = SQRT(dist);

	// Clamp to soften length.
	dist = max(dist, SOFTENING);
	distSqrt = max(distSqrt, SOFTENING_SQRT);

#if FASTMATH
	// This math is measurably faster, but its accuracy is questionable,
	// so we'll include it in the fast math preprocessor flag.
	float force = (massA * massB) / distSqrt / distSqrt / distSqrt;
	float3 ret = dir * force;
#else
	float force = massA * massB / dist;
	float3 ret = (dir * force) / distSqrt;
#endif

	return ret;
}


__kernel void CalcForceLocal(global Body* inBodies, int inBodiesLen, global MeshCell* inMesh, global int* meshNeighbors, const SimSettings sim, const SPHPreCalc sph)
{
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body and mesh cell from memory.
	MeshCell bodyCell = inMesh[(inBodies[(a)].MeshID)];
	MeshCell bodyCellParent = inMesh[bodyCell.ParentID];

	float3 iPos = (float3)(inBodies[(a)].PosX, inBodies[(a)].PosY, inBodies[(a)].PosZ);
	float iMass = inBodies[(a)].Mass;
	float3 iForce = (float3)(0.0f, 0.0f, 0.0f);
	float iDensity = 0.0f;
	float iPressure = 0.0f;

	// Resting Density.	
	iDensity = iMass * sph.fDensity;

	// *** Particle 2 Particle & SPH ***
	// Accumulate forces from all bodies within neighboring bottom level (local) cells. [THIS INCLUDES THE BODY'S OWN CELL]

	// Iterate parent cell neighbors.
	int start = bodyCellParent.NeighborStartIdx;
	int len = start + bodyCellParent.NeighborCount;
	for (int nc = start; nc < len; nc++)
	{
		// Iterate neighbor child cells.
		int nId = meshNeighbors[(nc)];
		int childStartIdx = inMesh[(nId)].ChildStartIdx;
		int childLen = childStartIdx + inMesh[(nId)].ChildCount;
		for (int c = childStartIdx; c < childLen; c++)
		{
			MeshCell cell = inMesh[(c)];

			// If the cell is far, compute force from cell.
			if (IsFar(bodyCell, cell))
			{
				float3 cellPos = (float3)(cell.CmX, cell.CmY, cell.CmZ);
				iForce += ComputeForce(cellPos, iPos, cell.Mass, iMass);
			}
			else // Otherwise compute force from cell bodies.
			{
				// Iterate the bodies within the cell.
				int mbStart = cell.BodyStartIdx;
				int mbLen = cell.BodyCount + mbStart;
				for (int mb = mbStart; mb < mbLen; mb++)
				{
					// Save us from ourselves.
					if (mb != a)
					{
						float jMass = inBodies[(mb)].Mass;
						float3 jPos = (float3)(inBodies[(mb)].PosX, inBodies[(mb)].PosY, inBodies[(mb)].PosZ);

						float3 dir = jPos - iPos;
						float dist = dot(dir, dir);
						float distSqrt = SQRT(dist);

						// If this body is within collision/SPH distance.
						if (distSqrt <= sph.kSize)
						{
							// Clamp SPH softening distance.
							dist = max(dist, SPH_SOFTENING);

							// Accumulate iDensity.
							float diff = sph.kSizeSq - dist;
							float fac = sph.fDensity * diff * diff * diff;
							iDensity += iMass * fac;
						}

						// Clamp gravity softening distance.
						distSqrt = max(distSqrt, SOFTENING_SQRT);

						// Accumulate body-to-body force.
#if FASTMATH
						// Faster but maybe less accurate.
						// Switch on preprocessor flag.
						float force = (jMass * iMass) / distSqrt / distSqrt / distSqrt;
						iForce += dir * force;
#else
						float force = jMass * iMass / dist;
						iForce += (dir * force) / distSqrt;
#endif
					}
				}
			}
		}
	}

	// Calculate pressure from density.
	iPressure = sim.GasK * iDensity;

	// Write back to memory.
	inBodies[(a)].ForceX = iForce.x;
	inBodies[(a)].ForceY = iForce.y;
	inBodies[(a)].ForceZ = iForce.z;
	inBodies[(a)].Density = iDensity;
	inBodies[(a)].Pressure = iPressure;
}

__kernel void CalcForceFar(global Body* inBodies, int inBodiesLen, global MeshCell* inMesh, global int* meshNeighbors, int meshTopStart, int meshTopEnd, global int* postNeeded)
{
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body and mesh cell from memory.
	float3 iPos = (float3)(inBodies[(a)].PosX, inBodies[(a)].PosY, inBodies[(a)].PosZ);
	float iMass = inBodies[(a)].Mass;
	float3 iForce = (float3)(inBodies[(a)].ForceX, inBodies[(a)].ForceY, inBodies[(a)].ForceZ);

	MeshCell bodyCell = inMesh[(inBodies[(a)].MeshID)];
	MeshCell bodyCellParent = inMesh[bodyCell.ParentID];

	// Set body cell to parent to move out of (or above?) the local field.
	bodyCell = bodyCellParent;

	// *** Particle 2 Mesh ***
	// Accumulate force from neighboring cells at each level.
	while (bodyCellParent.ParentID != -1)
	{
		// Get the next parent cell then iterate its neighbors.
		bodyCellParent = inMesh[(bodyCellParent.ParentID)];

		// Iterate parent cell neighbors.
		int start = bodyCellParent.NeighborStartIdx;
		int len = start + bodyCellParent.NeighborCount;
		for (int nc = start; nc < len; nc++)
		{
			// Iterate the child cells of the neighbor.
			int nId = meshNeighbors[(nc)];
			int childStartIdx = inMesh[(nId)].ChildStartIdx;
			int childLen = childStartIdx + inMesh[(nId)].ChildCount;
			for (int c = childStartIdx; c < childLen; c++)
			{
				// Accumulate force from the cells.
				MeshCell cell = inMesh[(c)];

				if (IsFar(bodyCell, cell))
				{
					float3 cellPos = (float3)(cell.CmX, cell.CmY, cell.CmZ);
					iForce += ComputeForce(cellPos, iPos, cell.Mass, iMass);
				}
			}
		}

		// Set body cell to parent in prep for next level.
		bodyCell = bodyCellParent;
	}

	// *** Particle 2 Mesh ***
	// Accumulate force from remaining distant cells at the top-most level.
	for (int top = meshTopStart; top < meshTopEnd; top++)
	{
		MeshCell cell = inMesh[(top)];

		if (IsFar(bodyCell, cell))
		{
			float3 cellPos = (float3)(cell.CmX, cell.CmY, cell.CmZ);
			iForce += ComputeForce(cellPos, iPos, cell.Mass, iMass);
		}
	}

	// Check for the phony roche condition.
	if (fast_length(iForce) > (iMass * 4.0f))
	{
		int iFlags = inBodies[(a)].Flag;
		int newFlags = SetFlag(iFlags, INROCHE, true);
		if (newFlags != iFlags)
		{
			iFlags = newFlags;
			postNeeded[0] = 1;
			inBodies[(a)].Flag = iFlags;
		}
	}

	// Write back to memory.
	inBodies[(a)].ForceX = iForce.x;
	inBodies[(a)].ForceY = iForce.y;
	inBodies[(a)].ForceZ = iForce.z;
}


// Is the specified cell a neighbor of the test cell?
bool IsFar(MeshCell cell, MeshCell testCell)
{
	if (abs(cell.IdxX - testCell.IdxX) > 1 || abs(cell.IdxY - testCell.IdxY) > 1 || abs(cell.IdxZ - testCell.IdxZ) > 1)
		return true;

	return false;
}

__kernel void ElasticCollisions(global Body* inBodies, int inBodiesLen, global MeshCell* inMesh, global int* meshNeighbors, int collisions, global int* postNeeded)
{
	// Get index for the current body.
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body from memory.
	Body outBody = inBodies[a];

	if (collisions == 1)
	{
		// Only proceed for bodies larger than 1 unit.
		if (outBody.Size <= 1.0f)
		{
			return;
		}

		// Get the current parent cell.
		MeshCell parentCell = inMesh[outBody.MeshID];
		int pcellSize = parentCell.Size;

		// Move up through parent cells until we find one
		// whose size is atleast as big as the target body.
		while (pcellSize < outBody.Size)
		{
			// Stop if we reach the top-most level.
			if (parentCell.ParentID == -1)
				break;

			parentCell = inMesh[parentCell.ParentID];
			pcellSize = parentCell.Size;
		}

		// Itereate the neighboring cells of the selected parent.
		for (int i = parentCell.NeighborStartIdx; i < parentCell.NeighborStartIdx + parentCell.NeighborCount; i++)
		{
			// Get the neighbor cell from the index.
			int nId = meshNeighbors[i];
			MeshCell nCell = inMesh[nId];

			// Iterate all the bodies within each neighboring cell.
			int mbStart = nCell.BodyStartIdx;
			int mbLen = nCell.BodyCount + mbStart;
			for (int mb = mbStart; mb < mbLen; mb++)
			{
				// Save us from ourselves.
				if (mb != a)
				{
					Body inBody = inBodies[mb];

					// Calc the distance and check for collision.
					float distX = outBody.PosX - inBody.PosX;
					float distY = outBody.PosY - inBody.PosY;
					float distZ = outBody.PosZ - inBody.PosZ;

					float dist = distX * distX + distY * distY + distZ * distZ;
					float distSqrt = (float)SQRT(dist);

					float colDist = outBody.Size * 0.5f + inBody.Size * 0.5f;
					if (distSqrt <= colDist)
					{
						// Calculate elastic collision forces.
						float colScale = (distX * (inBody.VeloX - outBody.VeloX) + distY * (inBody.VeloY - outBody.VeloY) + distZ * (inBody.VeloZ - outBody.VeloZ)) / dist;
						float forceX = distX * colScale;
						float forceY = distY * colScale;
						float forceZ = distZ * colScale;

						float colMass = inBody.Mass / (inBody.Mass + outBody.Mass);

						// If we're the bigger one, eat the other guy.
						if (outBody.Mass > inBody.Mass)
						{
							outBody = CollideBodies(outBody, inBody, colMass, forceX, forceY, forceZ);
							inBodies[a] = outBody;
							inBodies[mb] = SetFlagB(inBodies[mb], CULLED, true);
							postNeeded[0] = 1;

						}
						else if (outBody.Mass == inBody.Mass) // If we are the same size, use a different metric.
						{
							// Our UID is more gooder, eat the other guy.
							if (outBody.UID > inBody.UID)
							{
								outBody = CollideBodies(outBody, inBody, colMass, forceX, forceY, forceZ);
								inBodies[a] = outBody;
								inBodies[mb] = SetFlagB(inBodies[mb], CULLED, true);
								postNeeded[0] = 1;
							}
						}
					}
				}
			}
		}
	}
}

__kernel void SPHCollisions(global Body* inBodies, int inBodiesLen, global Body* outBodies, global MeshCell* inMesh, global int* meshNeighbors, global float3* centerMass, const SimSettings sim, const SPHPreCalc sph, global int* postNeeded)
{
	// Get index for the current body.
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body from memory.
	Body outBody = inBodies[(a)];
	float3 outPos = (float3)(outBody.PosX, outBody.PosY, outBody.PosZ);

	MeshCell bodyCell = inMesh[outBody.MeshID];

	if (sim.CollisionsOn == 1)
	{
		// Iterate parent cell neighbors.
		int start = inMesh[bodyCell.ParentID].NeighborStartIdx;
		int len = start + inMesh[bodyCell.ParentID].NeighborCount;
		for (int nc = start; nc < len; nc++)
		{
			// Iterate neighbor child cells.
			int nId = meshNeighbors[(nc)];
			int childStartIdx = inMesh[(nId)].ChildStartIdx;
			int childLen = childStartIdx + inMesh[(nId)].ChildCount;
			for (int c = childStartIdx; c < childLen; c++)
			{
				MeshCell cell = inMesh[(c)];

				// Check for close cell.
				if (!IsFar(bodyCell, cell))
				{
					int mbStart = cell.BodyStartIdx;
					int mbLen = mbStart + cell.BodyCount;

					// Iterate the neighbor cell bodies.
					for (int mb = mbStart; mb < mbLen; mb++)
					{
						// Double tests are bad.
						if (mb != a)
						{
							Body inBody = inBodies[(mb)];
							float3 inPos = (float3)(inBody.PosX, inBody.PosY, inBody.PosZ);
							float3 dir = outPos - inPos;
							float distSqrt = DISTANCE(outPos, inPos);

							// Calc the distance and check for collision.
							if (distSqrt <= sph.kSize)
							{
								//// Handle exact overlaps.
								//if (dist == 0)
								//{
								//	outBody.PosX += (outBody.UID + 1) * SPH_SOFTENING;
								//	outBody.PosY += (outBody.UID + 1) * SPH_SOFTENING;
								//	outBody.PosZ += (outBody.UID + 1) * SPH_SOFTENING;
								//}

								// Only do SPH collision if both bodies are in roche.
								// SPH collision.
								if (HasFlagB(outBody, INROCHE) && HasFlagB(inBody, INROCHE))
								{
									distSqrt = max(distSqrt, SPH_SOFTENING);
									float kDiff = sph.kSize - distSqrt;

									// Pressure force
									float pressScalar = outBody.Mass * (outBody.Pressure + inBody.Pressure) / (2.0f * outBody.Density);
									float pressGrad = sph.fPressure * kDiff * kDiff / distSqrt;

									outBody.ForceX += (dir.x * pressGrad) * pressScalar;
									outBody.ForceY += (dir.y * pressGrad) * pressScalar;
									outBody.ForceZ += (dir.z * pressGrad) * pressScalar;

									// Viscosity force
									float viscLaplace = sph.fViscosity * kDiff;
									float viscScalar = inBody.Mass * viscLaplace * sim.Viscosity / inBody.Density;

									outBody.ForceX += (inBody.VeloX - outBody.VeloX) * viscScalar;
									outBody.ForceY += (inBody.VeloY - outBody.VeloY) * viscScalar;
									outBody.ForceZ += (inBody.VeloZ - outBody.VeloZ) * viscScalar;
								}
							}
						}
					}
				}
			}
		}
	}

	// Integrate.
	outBody.VeloX += sim.DeltaTime * outBody.ForceX / outBody.Mass;
	outBody.VeloY += sim.DeltaTime * outBody.ForceY / outBody.Mass;
	outBody.VeloZ += sim.DeltaTime * outBody.ForceZ / outBody.Mass;

	outBody.PosX += sim.DeltaTime * outBody.VeloX;
	outBody.PosY += sim.DeltaTime * outBody.VeloY;
	outBody.PosZ += sim.DeltaTime * outBody.VeloZ;

	if (outBody.Lifetime > 0.0f)
	{
		outBody.Lifetime -= sim.DeltaTime * 4.0f;
	}

	// Check for and cull NANs.
	int nanCheck = isnan(outBody.PosX) + isnan(outBody.PosY) + isnan(outBody.PosZ);
	if (nanCheck > 0)
	{
		outBody.Flag = SetFlag(outBody.Flag, CULLED, true);
		postNeeded[0] = 1;
	}

	// Cull distant bodies.
	float3 cm = centerMass[0];
	float dist = DISTANCE(cm, (float3)(outBody.PosX, outBody.PosY, outBody.PosZ));

	if (dist > sim.CullDistance)
	{
		outBody.Flag = SetFlag(outBody.Flag, CULLED, true);
		postNeeded[0] = 1;
	}

	// Cull expired bodies.
	if (outBody.Lifetime < 0.0f && outBody.Lifetime > -100.0f)
	{
		outBody.Flag = SetFlag(outBody.Flag, CULLED, true);
		postNeeded[0] = 1;

		//printf("%s\n", "this is a test string\n");
	}

	// Write back to memory.
	outBodies[(a)] = outBody;
}

Body CollideBodies(Body bodyA, Body bodyB, float colMass, float forceX, float forceY, float forceZ)
{
	Body outBody = bodyA;

	outBody.VeloX += colMass * forceX;
	outBody.VeloY += colMass * forceY;
	outBody.VeloZ += colMass * forceZ;

	//// Don't increase size of black holes.
	//if (!HasFlagB(outBody, BLACKHOLE))
	//{
	//	float a1 = pow((outBody.Size * 0.5f), 2.0f);
	//	float a2 = pow((bodyB.Size * 0.5f), 2.0f);
	//	float area = a1 + a2;
	//	outBody.Size = (float)native_sqrt(area) * 2.0f;
	//}

	outBody.Mass += bodyB.Mass;

	return outBody;
}