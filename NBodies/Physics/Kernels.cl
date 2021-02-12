
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
	float Pressure;
	float Temp;
	float Lifetime;
	int MeshID;

} Body;

typedef struct
{
	int IdxX;
	int IdxY;
	int IdxZ;
	int NeighborStartIdx;
	int NeighborCount;
	int BodyStartIdx;
	int BodyCount;
	int ChildStartIdx;
	int ChildCount;
	float CmX;
	float CmY;
	float CmZ;
	float Mass;
	int Size;
	int ParentID;

} MeshCell;

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

// Bit twiddling, magic number, super duper morton computer...
// Credits: 
// https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
// https://graphics.stanford.edu/~seander/bithacks.html
long MortonNumber(long x, long y, long z)
{
	x &= 0x1fffff;
	x = (x | x << 32) & 0x1f00000000ffff;
	x = (x | x << 16) & 0x1f0000ff0000ff;
	x = (x | x << 8) & 0x100f00f00f00f00f;
	x = (x | x << 4) & 0x10c30c30c30c30c3;
	x = (x | x << 2) & 0x1249249249249249;

	y &= 0x1fffff;
	y = (y | y << 32) & 0x1f00000000ffff;
	y = (y | y << 16) & 0x1f0000ff0000ff;
	y = (y | y << 8) & 0x100f00f00f00f00f;
	y = (y | y << 4) & 0x10c30c30c30c30c3;
	y = (y | y << 2) & 0x1249249249249249;

	z &= 0x1fffff;
	z = (z | z << 32) & 0x1f00000000ffff;
	z = (z | z << 16) & 0x1f0000ff0000ff;
	z = (z | z << 8) & 0x100f00f00f00f00f;
	z = (z | z << 4) & 0x10c30c30c30c30c3;
	z = (z | z << 2) & 0x1249249249249249;

	return x | (y << 1) | (z << 2);
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


__kernel void ComputeMorts(global Body* bodies, int len, int padLen, int cellSizeExp, global long2* morts)
{
	int gid = get_global_id(0);

	if (gid < len)
	{
		int idxX = (int)floor(bodies[gid].PosX) >> cellSizeExp;
		int idxY = (int)floor(bodies[gid].PosY) >> cellSizeExp;
		int idxZ = (int)floor(bodies[gid].PosZ) >> cellSizeExp;

		long morton = MortonNumber(idxX, idxY, idxZ);

		morts[gid].x = morton;
		morts[gid].y = gid;
	}
	else
	{
		if (gid < padLen)
		{
			// Fill padded region...
			morts[gid].x = LONG_MAX;
			morts[gid].y = PADDING_ELEM;
		}
	}
}


// Builds initial cell map from storted morton numbers.
__kernel void MapMorts(global long* morts, int len, global int* cellmap, global int* counts, volatile __local int* lMap, int threads, int step)
{
	int gid = get_global_id(0);
	int tid = get_local_id(0);
	int bid = get_group_id(0);

	if (gid >= len)
		return;

	// Local count.
	volatile __local int lCount;

	// First thread initializes local count.
	if (tid == 0)
		lCount = 0;

	// Init local map.
	lMap[tid] = -1;

	// Sync local threads.
	barrier(CLK_LOCAL_MEM_FENCE);

	// Apply the step size to the gid.
	// Step size of 1 for long type, 2 for long2 type inputs.
	int gidOff = gid * step;

	// Compare two morton numbers, record the location and increment count if they dont match.
	// This is where a new cell starts and the previous cell ends.
	if ((gid + 1) < len && morts[gidOff] != morts[gidOff + step])
	{
		atomic_inc(&lCount);
		lMap[tid] = gid + 1;
	}

	// Sync local threads.
	barrier(CLK_LOCAL_MEM_FENCE);

	// Finally, the first thread dumps and packs the local memory for the block.
	if (tid == 0)
	{
		// Write the found indexes for this block to its location in global memory.
		int n = 0;
		for (int i = 0; i < threads; i++)
		{
			int val = lMap[i];

			if (val > -1)
				cellmap[threads * bid + n++] = val;
		}

		// Write the cell count for the block to global memory.
		counts[bid] = lCount;
	}
}

// Compresses/packs initial cell maps into the beginning of the buffer.
// N threads = N blocks used in the map kernel.
__kernel void CompressMap(int len, global int* cellmapIn, global int* cellmapOut, global int* counts, int threads)
{
	int gid = get_global_id(0);

	if (gid >= len)
		return;

	int rStart = gid * threads; // Read location. Offset by block size.
	int nCount = counts[gid]; // Number of items this thread will copy.
	int wStart = 0; // Write location into completed map.

	// Find the write location for this thread.
	if (gid > 0)
	{
		for (int b = 0; b < gid; b++)
			wStart += counts[b];
	}

	// Copy the indexes to the output at the correct location.
	for (int i = 0; i < nCount; i++)
		cellmapOut[wStart + i] = cellmapIn[rStart + i];
}


// Read indexes from the sorted morton buffer and copy bodies to their sorted location.
__kernel void ReindexBodies(global Body* inBodies, int blen, global long2* sortMap, global Body* outBodies)
{
	int b = get_global_id(0);

	if (b >= blen)
		return;

	int newIdx = (int)sortMap[b].y;

	// Make sure we don't hit a padded element.
	// This condition may be safe to remove.
	//if (newIdx > -1)
	outBodies[b] = inBodies[newIdx];
}


__kernel void BuildBottom(global Body* inBodies, global MeshCell* mesh, int meshLen, int bodyLen, global int* cellMap, int cellSizeExp, int cellSize, global long* parentMorts)
{
	int m = get_global_id(0);

	if (m >= meshLen)
		return;

	int firstIdx = 0;
	if (m > 0)
		firstIdx = cellMap[m - 1];

	int lastIdx = cellMap[m];
	if (m == meshLen - 1)
		lastIdx = bodyLen;

	float fPosX = inBodies[firstIdx].PosX;
	float fPosY = inBodies[firstIdx].PosY;
	float fPosZ = inBodies[firstIdx].PosZ;
	float fMass = inBodies[firstIdx].Mass;

	double3 nCM = (double3)(fMass * fPosX, fMass * fPosY, fMass * fPosZ);
	double nMass = fMass;

	MeshCell newCell;
	newCell.IdxX = (int)floor(fPosX) >> cellSizeExp;
	newCell.IdxY = (int)floor(fPosY) >> cellSizeExp;
	newCell.IdxZ = (int)floor(fPosZ) >> cellSizeExp;
	newCell.Size = cellSize;
	newCell.BodyStartIdx = firstIdx;
	newCell.BodyCount = 1;
	newCell.NeighborStartIdx = -1;
	newCell.NeighborCount = 0;
	newCell.ChildStartIdx = -1;
	newCell.ChildCount = 0;
	newCell.ParentID = -1;

	// Compute parent level morton numbers.
	int idxX = newCell.IdxX >> 1;
	int idxY = newCell.IdxY >> 1;
	int idxZ = newCell.IdxZ >> 1;
	long morton = MortonNumber(idxX, idxY, idxZ);
	parentMorts[m] = morton;

	inBodies[firstIdx].MeshID = m;

	for (int i = firstIdx + 1; i < lastIdx; i++)
	{
		float posX = inBodies[i].PosX;
		float posY = inBodies[i].PosY;
		float posZ = inBodies[i].PosZ;
		float mass = inBodies[i].Mass;

		nMass += mass;
		nCM.x += mass * posX;
		nCM.y += mass * posY;
		nCM.z += mass * posZ;

		newCell.BodyCount++;

		inBodies[i].MeshID = m;
	}

	newCell.Mass = nMass;
	newCell.CmX = (nCM.x / nMass);
	newCell.CmY = (nCM.y / nMass);
	newCell.CmZ = (nCM.z / nMass);

	mesh[m] = newCell;
}


__kernel void BuildTop(global MeshCell* mesh, int parentLen, int childsStart, int childsEnd, global int* cellMap, int cellSize, int level, global long* parentMorts)
{
	int m = get_global_id(0);

	if (m >= parentLen)
		return;

	int newIdx = m + childsEnd;

	int firstIdx = childsStart;
	if (m > 0)
		firstIdx += cellMap[m - 1];

	int lastIdx = childsStart + cellMap[m];
	if (m == parentLen - 1)
		lastIdx = childsEnd;

	double3 nCM;
	double nMass;

	MeshCell newCell;
	newCell.IdxX = mesh[firstIdx].IdxX >> 1;
	newCell.IdxY = mesh[firstIdx].IdxY >> 1;
	newCell.IdxZ = mesh[firstIdx].IdxZ >> 1;

	nMass = mesh[firstIdx].Mass;
	nCM = (double3)(nMass * mesh[firstIdx].CmX, nMass * mesh[firstIdx].CmY, nMass * mesh[firstIdx].CmZ);

	newCell.Size = cellSize;
	newCell.BodyStartIdx = mesh[firstIdx].BodyStartIdx;
	newCell.BodyCount = mesh[firstIdx].BodyCount;
	newCell.NeighborStartIdx = -1;
	newCell.NeighborCount = 0;
	newCell.ChildStartIdx = firstIdx;
	newCell.ChildCount = 1;
	newCell.ParentID = -1;

	mesh[firstIdx].ParentID = newIdx;

	// Compute parent level morton numbers.
	int idxX = newCell.IdxX >> 1;
	int idxY = newCell.IdxY >> 1;
	int idxZ = newCell.IdxZ >> 1;
	long morton = MortonNumber(idxX, idxY, idxZ);
	parentMorts[m] = morton;


	for (int i = firstIdx + 1; i < lastIdx; i++)
	{
		float mass = mesh[i].Mass;

		nMass += mass;
		nCM.x += mass * mesh[i].CmX;
		nCM.y += mass * mesh[i].CmY;
		nCM.z += mass * mesh[i].CmZ;

		newCell.ChildCount++;
		newCell.BodyCount += mesh[i].BodyCount;

		mesh[i].ParentID = newIdx;
	}

	newCell.Mass = nMass;
	newCell.CmX = (nCM.x / nMass);
	newCell.CmY = (nCM.y / nMass);
	newCell.CmZ = (nCM.z / nMass);

	mesh[newIdx] = newCell;
}


// Top-down mesh based nearest neighbor search.
__kernel void BuildNeighborsMesh(global MeshCell* mesh, global int* neighborIndex, int botOffset, int levels, int level, int start, int end)
{
	int m = get_global_id(0);
	int readM = m + start;

	if (readM >= end)
		return;

	// Write location of the neighbor list.
	long offset = (readM - botOffset) * 27;
	int count = 0;

	MeshCell cell = mesh[readM];

	if (level == levels)
	{
		// For the first pass, iterate all top-level cells and brute force the neighbors. (There won't be many, so this is fast.)
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
	else
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

	mesh[readM].NeighborStartIdx = offset;
	mesh[readM].NeighborCount = count;
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

#if FASTMATH
	float dist = dot(dir, dir);
	float distSqrt = SQRT(dist);
#else
	float distSqrt = LENGTH(dir);
	float dist = pow(distSqrt, 2);
#endif

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


__kernel void CalcForce(global Body* inBodies, int inBodiesLen, global MeshCell* inMesh, global int* meshNeighbors, const SimSettings sim, const SPHPreCalc sph, int meshTopStart, int meshTopEnd, global int* postNeeded)
{
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy body pos & mass.
	float3 iPos = (float3)(inBodies[(a)].PosX, inBodies[(a)].PosY, inBodies[(a)].PosZ);
	float iMass = inBodies[(a)].Mass;

	// Copy body mesh cell.
	MeshCell bodyCell = inMesh[(inBodies[(a)].MeshID)];

	float3 iForce = (float3)(0.0f, 0.0f, 0.0f);
	float iDensity = 0.0f;
	float iPressure = 0.0f;

	// Resting Density.	
	iDensity = iMass * sph.fDensity;

	// *** Particle 2 Particle/Mesh & SPH ***
	// Walk the mesh tree and accumulate forces from all bodies & cells within the local region. [THIS INCLUDES THE BODY'S OWN CELL]
	bool done = false;
	bool bottom = true;
	while (!done)
	{
		MeshCell bodyCellParent = inMesh[bodyCell.ParentID];

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
				bool far = IsFar(bodyCell, cell);
				if (far)
				{
					float3 cellPos = (float3)(cell.CmX, cell.CmY, cell.CmZ);
					iForce += ComputeForce(cellPos, iPos, cell.Mass, iMass);
				}
				else if (bottom && !far) // Otherwise compute force from cell bodies if we are on the bottom level.
				{
					// Iterate the bodies within the cell.
					int mbStart = cell.BodyStartIdx;
					int mbLen = cell.BodyCount + mbStart;
					for (int mb = mbStart; mb < mbLen; mb++)
					{
						// Save us from ourselves.
						if (mb != a)
						{
							float3 jPos = (float3)(inBodies[(mb)].PosX, inBodies[(mb)].PosY, inBodies[(mb)].PosZ);
							float jMass = inBodies[(mb)].Mass;
							float3 dir = jPos - iPos;

#if FASTMATH
							float dist = dot(dir, dir);
							float distSqrt = SQRT(dist);
#else
							float distSqrt = LENGTH(dir);
							float dist = pow(distSqrt, 2);
#endif

							// If this body is within collision/SPH distance.
							if (distSqrt <= sph.kSize)
							{
								// Accumulate iDensity.
								float diff = sph.kSizeSq - max(dist, SPH_SOFTENING);
								float fac = sph.fDensity * diff * diff * diff;
								iDensity += iMass * fac;
							}

							// Clamp gravity softening distance.
							distSqrt = max(distSqrt, SOFTENING_SQRT);
							dist = max(dist, SOFTENING);

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

		bodyCell = bodyCellParent;
		done = (bodyCell.ParentID == -1);
		bottom = false;
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

	// Calculate pressure from density.
	iPressure = sim.GasK * iDensity;

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
	inBodies[(a)].Density = iDensity;
	inBodies[(a)].Pressure = iPressure;
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

	float maxTemp = 10000.0f;//4000.0f
	double tempDelta = 0;
	int neighbors = 0;

	// Copy current body from memory.
	Body outBody = inBodies[(a)];
	float3 outPos = (float3)(outBody.PosX, outBody.PosY, outBody.PosZ);

	MeshCell bodyCell = inMesh[outBody.MeshID];

	if (sim.CollisionsOn == 1 && HasFlagB(outBody, INROCHE))
	{
		// Iterate parent cell neighbors.
		int start = inMesh[bodyCell.ParentID].NeighborStartIdx;
		int len = start + inMesh[bodyCell.ParentID].NeighborCount;

		// PERF HACK: Mask out the len for bodies at resting density to skip the tree walk?
		len = len * !((outBody.Mass * sph.fDensity) == outBody.Density);

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
					// Record # of neighbors encountered.
					neighbors++;

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

#if FASTMATH
							float dist = dot(dir, dir);
							float distSqrt = SQRT(dist);
#else
							float distSqrt = LENGTH(dir);
							float dist = pow(distSqrt, 2);
#endif

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

								// Clamp the dist to the SPH softening value.
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

								float veloDiffX = inBody.VeloX - outBody.VeloX;
								float veloDiffY = inBody.VeloY - outBody.VeloY;
								float veloDiffZ = inBody.VeloZ - outBody.VeloZ;

								outBody.ForceX += veloDiffX * viscScalar;
								outBody.ForceY += veloDiffY * viscScalar;
								outBody.ForceZ += veloDiffZ * viscScalar;

								float velo = veloDiffX * veloDiffX + veloDiffY * veloDiffY + veloDiffZ * veloDiffZ;

								// Temp delta from p2p conduction.
								float tempK = 0.5f;//0.5f;//2.0f;//1.0f;
								float tempDiff = outBody.Temp - inBody.Temp;
								tempDelta += (-0.5 * tempK) * (tempDiff / dist);

								// Temp delta from p2p friction.
								float coeff = 0.0005f;//0.0004f;
								float adhesion = 0.1f; //viscosity;//0.1f;
								float heatJ = 8.31f;
								float heatFac = 1950;
								float factor = 0.0975f;
								tempDelta += (factor * velo) / heatJ;

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

	outBody.Temp += tempDelta * sim.DeltaTime;

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


	// Black body radiation.
	//double SBC = 0.000000056703;
	double SBC = 0.056703;

	// Only apply black body radiation if we have a temp, and we are not inside a clump. ( neighbors < 27 )
	if (outBody.Temp > 0 && neighbors < 27)
	{
		//double lossD = SBC * pow((double)outBody.Temp, 4.0) * 0.5f;//0.1;//1.0;//0.785;
		double lossD = SBC * pow((double)outBody.Temp, 2.0) * 0.05f;//0.1;//1.0;//0.785;
		outBody.Temp -= ((lossD / (double)(outBody.Mass * 0.25)) / 4.186) * sim.DeltaTime;
	}

	outBody.Temp = min(outBody.Temp, maxTemp);
	outBody.Temp = max(outBody.Temp, 1.0f);


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



// *** SORTING KERNELS ***

// Sort kernels
// EB Jun 2011
//
// Credit & Thanks to:
// Eric Bainville - OpenCL Sorting
// http://www.bealto.com/gpu-sorting_intro.html
//

#define KTYPE long

typedef long2 data_t;
#define getKey(a) ((a).x)
#define getValue(a) ((a).y)
#define makeData(k,v) ((long2)((k),(v)))

// One thread per record
__kernel void Copy(__global const data_t* in, __global data_t* out)
{
	int i = get_global_id(0); // current thread
	out[i] = in[i]; // copy
}

// Added 'getValue(b) == PADDING_ELEM' to force padding elements to the end.
#define ORDER(a,b) { bool swap = reverse ^ (getKey(a)<getKey(b) || getValue(b) == PADDING_ELEM); data_t auxa = a; data_t auxb = b; a = (swap)?auxb:auxa; b = (swap)?auxa:auxb; }


// N/2 threads
__kernel void ParallelBitonic_B2(__global data_t* data, int inc, int dir)
{
	int t = get_global_id(0); // thread index
	int low = t & (inc - 1); // low order bits (below INC)
	int i = (t << 1) - low; // insert 0 at position INC
	bool reverse = ((dir & i) == 0); // asc/desc order
	data += i; // translate to first value

			   // Load
	data_t x0 = data[0];
	data_t x1 = data[inc];

	// Sort
	ORDER(x0, x1)

		// Store
		data[0] = x0;
	data[inc] = x1;
}

// N/4 threads
__kernel void ParallelBitonic_B4(__global data_t* data, int inc, int dir)
{
	inc >>= 1;
	int t = get_global_id(0); // thread index
	int low = t & (inc - 1); // low order bits (below INC)
	int i = ((t - low) << 2) + low; // insert 00 at position INC
	bool reverse = ((dir & i) == 0); // asc/desc order
	data += i; // translate to first value

			   // Load
	data_t x0 = data[0];
	data_t x1 = data[inc];
	data_t x2 = data[2 * inc];
	data_t x3 = data[3 * inc];

	// Sort
	ORDER(x0, x2)
		ORDER(x1, x3)
		ORDER(x0, x1)
		ORDER(x2, x3)

		// Store
		data[0] = x0;
	data[inc] = x1;
	data[2 * inc] = x2;
	data[3 * inc] = x3;
}

// Added 'getValue(b) == PADDING_ELEM' to force padding elements to the end.
#define ORDERV(x,a,b) { bool swap = reverse ^ (getKey(x[a])<getKey(x[b]) || getValue(x[b]) == PADDING_ELEM); \
      data_t auxa = x[a]; data_t auxb = x[b]; \
      x[a] = (swap)?auxb:auxa; x[b] = (swap)?auxa:auxb; }

#define B2V(x,a) { ORDERV(x,a,a+1) }
#define B4V(x,a) { for (int i4=0;i4<2;i4++) { ORDERV(x,a+i4,a+i4+2) } B2V(x,a) B2V(x,a+2) }
#define B8V(x,a) { for (int i8=0;i8<4;i8++) { ORDERV(x,a+i8,a+i8+4) } B4V(x,a) B4V(x,a+4) }
#define B16V(x,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,a+i16,a+i16+8) } B8V(x,a) B8V(x,a+8) }

// N/8 threads
__kernel void ParallelBitonic_B8(__global data_t* data, int inc, int dir)
{
	inc >>= 2;
	int t = get_global_id(0); // thread index
	int low = t & (inc - 1); // low order bits (below INC)
	int i = ((t - low) << 3) + low; // insert 000 at position INC
	bool reverse = ((dir & i) == 0); // asc/desc order
	data += i; // translate to first value

			   // Load
	data_t x[8];
	for (int k = 0; k < 8; k++) x[k] = data[k * inc];

	// Sort
	B8V(x, 0)

		// Store
		for (int k = 0; k < 8; k++) data[k * inc] = x[k];
}

// N/16 threads
__kernel void ParallelBitonic_B16(__global data_t* data, int inc, int dir)
{
	inc >>= 3;
	int t = get_global_id(0); // thread index
	int low = t & (inc - 1); // low order bits (below INC)
	int i = ((t - low) << 4) + low; // insert 0000 at position INC
	bool reverse = ((dir & i) == 0); // asc/desc order
	data += i; // translate to first value

			   // Load
	data_t x[16];
	for (int k = 0; k < 16; k++) x[k] = data[k * inc];

	// Sort
	B16V(x, 0)

		// Store
		for (int k = 0; k < 16; k++) data[k * inc] = x[k];
}


// N/2 threads, AUX[2*WG]
__kernel void ParallelBitonic_C2(__global data_t* data, int inc0, int dir, __local data_t* aux)
{
	int t = get_global_id(0); // thread index
	int wgBits = 2 * get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 2*WG)

	for (int inc = inc0; inc > 0; inc >>= 1)
	{
		int low = t & (inc - 1); // low order bits (below INC)
		int i = (t << 1) - low; // insert 0 at position INC
		bool reverse = ((dir & i) == 0); // asc/desc order
		data_t x0, x1;

		// Load
		if (inc == inc0)
		{
			// First iteration: load from global memory
			x0 = data[i];
			x1 = data[i + inc];
		}
		else
		{
			// Other iterations: load from local memory
			barrier(CLK_LOCAL_MEM_FENCE);
			x0 = aux[i & wgBits];
			x1 = aux[(i + inc) & wgBits];
		}

		// Sort
		ORDER(x0, x1)

			//	printf("%i : %i \n", x0.x, x0.y);

			// Store
			if (inc == 1)
			{
				// Last iteration: store to global memory
				data[i] = x0;
				data[i + inc] = x1;
			}
			else
			{
				// Other iterations: store to local memory
				barrier(CLK_LOCAL_MEM_FENCE);
				aux[i & wgBits] = x0;
				aux[(i + inc) & wgBits] = x1;
			}
	}
}

__kernel void ParallelBitonic_C4(__global data_t* data, int inc0, int dir, __local data_t* aux)
{
	int t = get_global_id(0); // thread index
	int wgBits = 4 * get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 4*WG)
	int inc, low, i;
	bool reverse;
	data_t x[4];

	// First iteration, global input, local output
	inc = inc0 >> 1;
	low = t & (inc - 1); // low order bits (below INC)
	i = ((t - low) << 2) + low; // insert 00 at position INC
	reverse = ((dir & i) == 0); // asc/desc order
	for (int k = 0; k < 4; k++) x[k] = data[i + k * inc];
	B4V(x, 0);

	//printf("%i : %i \n", x[0].x, x[0].y);

	for (int k = 0; k < 4; k++) aux[(i + k * inc) & wgBits] = x[k];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Internal iterations, local input and output
	for (; inc > 1; inc >>= 2)
	{
		low = t & (inc - 1); // low order bits (below INC)
		i = ((t - low) << 2) + low; // insert 00 at position INC
		reverse = ((dir & i) == 0); // asc/desc order
		for (int k = 0; k < 4; k++) x[k] = aux[(i + k * inc) & wgBits];
		B4V(x, 0);
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < 4; k++) aux[(i + k * inc) & wgBits] = x[k];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Final iteration, local input, global output, INC=1
	i = t << 2;
	reverse = ((dir & i) == 0); // asc/desc order
	for (int k = 0; k < 4; k++) x[k] = aux[(i + k) & wgBits];
	B4V(x, 0);
	for (int k = 0; k < 4; k++) data[i + k] = x[k];
}
