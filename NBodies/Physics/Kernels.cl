
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
	float Pressure;
	float Lifetime;
	int MeshID;

} Body;

typedef struct
{
	int IdxX;
	int IdxY;
	int NeighborStartIdx;
	int NeighborCount;
	int BodyStartIdx;
	int BodyCount;
	int ChildStartIdx;
	int ChildCount;
	float Mass;
	float CmX;
	float CmY;
	int Size;
	int ParentID;
	int Level;

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

// Lookup table for neighbor search.
constant int2 N_OFFSET_LUT[] = { { -1,-1 }, { 0,-1 }, { 1,-1 }, { -1,0 }, { 1,0 }, { -1,1 }, { 0,1 }, { 1,1 } };

#define DISTANCE(a,b) distance(a,b)
#define SQRT(a) sqrt(a)
#define LENGTH(a) length(a)

//#define DISTANCE(a,b) fast_distance(a,b)
//#define SQRT(a) half_sqrt(a)
//#define LENGTH(a) fast_length(a)

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

int BlockCount(int len, int threads)
{
	int blocks = len / threads;
	int mod = len % threads;

	if (mod > 0)
		blocks += 1;

	return blocks;
}

// Bit twiddling, magic number, super duper morton computer...
// Credits: 
// https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
// https://graphics.stanford.edu/~seander/bithacks.html
long MortonNumberInt2(int2 idx)
{
	long x = idx.x;
	long y = idx.y;

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

	return x | (y << 1);
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

			float dist = (distX * distX) + (distY * distY);
			float colDist = aRad + bRad;

			if (dist <= (colDist * colDist))
			{
				float distSqrt = (float)native_sqrt(dist);
				float overlap = 0.5f * (distSqrt - aRad - bRad);

				bodyA.PosX -= overlap * (distX) / distSqrt;
				bodyA.PosY -= overlap * (distY) / distSqrt;
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
		int2 idx = (int2)((int)floor(bodies[gid].PosX), (int)floor(bodies[gid].PosY));
		long morton = MortonNumberInt2(idx >> cellSizeExp);

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
__kernel void MapMorts(global long* morts, int len, global int* cellmap, global int* counts, volatile __local int* lMap, int step, int level, global int* levelCounts, long bufLen)
{
	int gid = get_global_id(0);
	int tid = get_local_id(0);
	int bid = get_group_id(0);

	if (len < 0)
		len = levelCounts[level - 1];

	if (gid >= bufLen)
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

	// Compare two morton numbers, and record the location if they dont match.
	// This is where a new cell starts and the previous cell ends.
	if ((gid + 1) < len && morts[gidOff] != morts[gidOff + step])
	{
		atomic_inc(&lCount);
		lMap[tid] = gid + 1;
	}

	// Sync local threads.
	barrier(CLK_LOCAL_MEM_FENCE);

	// All threads write their results back to global memory.
	cellmap[gid] = lMap[tid];

	// Finally, the first thread writes the cell count to global memory.
	if (tid == 0)
		counts[bid] = lCount;
}


// Compresses/packs initial cell maps into the beginning of the buffer.
// N threads = N blocks used in the map kernel.
__kernel void CompressMap(int blocks, global int* cellmapIn, global int* cellmapOut, global int* counts, global int* levelCounts, global int* levelIdx, int level)
{
	int gid = get_global_id(0);
	int threads = get_local_size(0);
	int len = 0;

	if (blocks > 0)
		len = blocks;
	else
		len = BlockCount(levelCounts[level - 1], threads);

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

	// Write found values to global memory.
	int n = 0;
	for (int i = 0; i < threads; i++)
	{
		int inVal = cellmapIn[rStart + i];

		if (inVal > -1)
			cellmapOut[wStart + n++] = inVal;
	}

	// Use the last thread to populate level count & level indexes.
	if (gid == len - 1)
	{
		int tCount = wStart + nCount + 1;
		levelCounts[level] = tCount;
		levelIdx[level + 1] = levelIdx[level] + tCount;
	}
}


// Read indexes from the sorted morton buffer and copy bodies to their sorted location.
__kernel void ReindexBodies(global Body* inBodies, int blen, global long2* sortMap, global Body* outBodies)
{
	int b = get_global_id(0);

	if (b >= blen)
		return;

	int newIdx = (int)sortMap[b].y;

	outBodies[b] = inBodies[newIdx];
}


__kernel void BuildBottom(global Body* inBodies, global int2* meshIdxs, global int2* meshBodyBounds, global float4* meshCMM, global int4* meshSPL, global int* levelCounts, int bodyLen, global int* cellMap, int cellSizeExp, int cellSize, global long* parentMorts, long bufLen)
{
	int m = get_global_id(0);

	int meshLen = levelCounts[0];

	if (m >= meshLen || m >= bufLen)
		return;

	int firstIdx = 0;
	if (m > 0)
		firstIdx = cellMap[m - 1];

	int lastIdx = cellMap[m];
	if (m == meshLen - 1)
		lastIdx = bodyLen;

	float fPosX = inBodies[firstIdx].PosX;
	float fPosY = inBodies[firstIdx].PosY;
	float fMass = inBodies[firstIdx].Mass;

	double2 nCM = (double2)(fMass * fPosX, fMass * fPosY);
	double nMass = (double)fMass;

	int2 meshIdx = (int2)((int)floor(fPosX) >> cellSizeExp, (int)floor(fPosY) >> cellSizeExp);
	meshIdxs[m] = meshIdx;
	meshBodyBounds[m] = (int2)(firstIdx, lastIdx - firstIdx);
	meshSPL[m] = (int4)(cellSize, -1, 0, 0);

	// Compute parent level morton numbers.
	long morton = MortonNumberInt2(meshIdx >> 1);
	parentMorts[m] = morton;

	inBodies[firstIdx].MeshID = m;

	for (int i = firstIdx + 1; i < lastIdx; i++)
	{
		float posX = inBodies[i].PosX;
		float posY = inBodies[i].PosY;
		float mass = inBodies[i].Mass;

		nMass += mass;
		nCM.x += mass * posX;
		nCM.y += mass * posY;

		inBodies[i].MeshID = m;
	}

	meshCMM[m] = (float4)((nCM.x / nMass), (nCM.y / nMass), nMass, 0);
}


__kernel void BuildTop(global int2* meshIdxs, global int2* meshBodyBounds, global int2* meshChildBounds, global float4* meshCMM, global int4* meshSPL, global int* levelCounts, global int* levelIdx, global int* cellMap, int cellSize, int level, global long* parentMorts, long bufLen)
{
	int m = get_global_id(0);

	int parentLen = levelCounts[level];

	if (m >= parentLen || m >= bufLen)
		return;

	int childsStart = levelIdx[level - 1];
	int childsEnd = levelIdx[level];

	int newIdx = m + childsEnd;

	if (newIdx >= bufLen)
		return;

	int firstIdx = childsStart;
	int lastIdx = childsStart + cellMap[m];

	if (m > 0)
		firstIdx += cellMap[m - 1];

	if (m == parentLen - 1)
		lastIdx = childsEnd;

	double2 nCM;
	double nMass;

	int2 firstMIdx = meshIdxs[firstIdx];
	float4 firstCMM = meshCMM[firstIdx];
	int2 bodyBounds = meshBodyBounds[firstIdx];

	int2 meshIdx = firstMIdx >> 1;
	int2 childBounds = (int2)(firstIdx, 1);

	nMass = (double)firstCMM.z;
	nCM = (double2)(nMass * firstCMM.x, nMass * firstCMM.y);

	// Compute parent level morton numbers.
	long morton = MortonNumberInt2(meshIdx >> 1);
	parentMorts[m] = morton;

	meshSPL[firstIdx].y = newIdx;

	for (int i = firstIdx + 1; i < lastIdx; i++)
	{
		bodyBounds.y += meshBodyBounds[i].y;
		childBounds.y++;

		float4 childCMM = meshCMM[i];
		float mass = childCMM.z;
		nMass += mass;
		nCM.x += mass * childCMM.x;
		nCM.y += mass * childCMM.y;

		meshSPL[i].y = newIdx;
	}

	meshIdxs[newIdx] = meshIdx;
	meshSPL[newIdx] = (int4)(cellSize, -1, level, 0);
	meshChildBounds[newIdx] = childBounds;
	meshBodyBounds[newIdx] = bodyBounds;
	meshCMM[newIdx] = (float4)((nCM.x / nMass), (nCM.y / nMass), nMass, 0);
}


// Top-down mesh based nearest neighbor search.
__kernel void BuildNeighborsMesh(global int2* meshIdxs, global int4* meshSPL, global int2* meshNBounds, global int2* meshChildBounds, global int* neighborIndex, int botOffset, int levels, int level, int start, int end)
{
	int m = get_global_id(0);
	int readM = m + start;

	if (readM >= end)
		return;

	// Write location of the neighbor list.
	long offset = (readM - botOffset) * 9;
	int count = 0;

	int2 cellIdxs = meshIdxs[readM];
	int cellParent = meshSPL[readM].y;

	if (level == levels)
	{
		// If there is a large number of top level cells, use a binary search strategy.
		if ((end - start) > 1000)
		{
			// Set the first neighbor.
			neighborIndex[offset + count++] = readM;

			// Find the remaining neighbors.
#pragma unroll 8
			for (int i = 0; i < 8; i++)
			{
				// Perform a binary search.
				int2 offsetIdx = cellIdxs + N_OFFSET_LUT[i];
				int foundIdx = BinarySearch(meshIdxs, offsetIdx, start, end);

				if (foundIdx > -1)
					neighborIndex[offset + count++] = foundIdx;
			}
		}
		else // Otherwise just brute force all of them.
		{
			// For the first pass, iterate all top-level cells and brute force the neighbors. (There won't be many, so this is fast.)
			for (int i = start; i < end; i++)
			{
				int2 checkIdxs = meshIdxs[i];

				if (!IsFar(cellIdxs, checkIdxs))
				{
					neighborIndex[(offset + count++)] = i;
				}
			}
		}
	}
	else
	{
		// Use the mesh tree hierarchy & neighbors found for parent level to narrow down the search area significantly.
		int2 parentNBounds = meshNBounds[cellParent];

		// Iterate parent cell neighbors.
		int pstart = parentNBounds.x;
		int plen = pstart + parentNBounds.y;
		for (int nc = pstart; nc < plen; nc++)
		{
			// Iterate neighbor child cells.
			int nId = neighborIndex[(nc)];
			int2 childBounds = meshChildBounds[nId];

			int childStartIdx = childBounds.x;
			int childLen = childStartIdx + childBounds.y;
			for (int c = childStartIdx; c < childLen; c++)
			{
				int2 childIdxs = meshIdxs[c];

				// Check for neighbors and add them to the list.
				if (!IsFar(cellIdxs, childIdxs))
				{
					neighborIndex[(offset + count++)] = c;
				}
			}
		}
	}

	meshNBounds[readM] = (int2)(offset, count);
}


// All at once binary nearest neighbor search.
__kernel void BuildNeighborsBinary(global int2* meshIdxs, global int2* meshNBounds, global int4* meshSPL, global int* neighborIndex, global int* levelIdx, int len, int botOffset)
{
	int gid = get_global_id(0);

	if (gid >= len)
		return;

	int meshIdx = gid + botOffset;
	int2 cellIdx = meshIdxs[meshIdx];
	int cellLevel = meshSPL[meshIdx].z;
	int start = levelIdx[cellLevel];
	int end = levelIdx[cellLevel + 1] - 1;
	int neighborIdx = gid * 9;
	int count = 0;

	// Set the first neighbor.
	neighborIndex[neighborIdx + count++] = meshIdx;

	// Find the remaining neighbors.
#pragma unroll 8
	for (int i = 0; i < 8; i++)
	{
		int2 offsetIdx = cellIdx + N_OFFSET_LUT[i];
		int foundIdx = BinarySearch(meshIdxs, offsetIdx, start, end);

		if (foundIdx > -1)
			neighborIndex[neighborIdx + count++] = foundIdx;
	}

	meshNBounds[meshIdx] = (int2)(neighborIdx, count);
}


int BinarySearch(global int2* meshIdxs, int2 cellIdx, int start, int end)
{
	int lo = start;
	int hi = end;

	long key = MortonNumberInt2(cellIdx);
	int idx = -1;
	int foundIdx = -1;

	while (lo <= hi)
	{
		idx = lo + ((hi - lo) >> 1);
		long testKey = MortonNumberInt2(meshIdxs[idx]);

		if (key == testKey)
		{
			foundIdx = idx;
		}

		bool right = key < testKey;
		hi = select(hi, (idx - 1), right);
		lo = select((idx + 1), lo, right);
	}

	return foundIdx;
}


__kernel void CalcCenterOfMass(global float4* meshCMM, global float2* cm, int start, int end)
{
	double cmX = 0;
	double cmY = 0;
	double mass = 0;

	for (int i = start; i < end; i++)
	{
		float4 cellCMM = meshCMM[i];

		mass += cellCMM.z;
		cmX += cellCMM.z * cellCMM.x;
		cmY += cellCMM.z * cellCMM.y;

	}

	cmX = cmX / mass;
	cmY = cmY / mass;

	cm[0] = (float2)(cmX, cmY);
}


float2 CellForce(float2 posA, float2 posB, float massA, float massB)
{
	float2 dir = posA - posB;
	float dist = dot(dir, dir);
	float distSqrt = SQRT(dist);

	// Clamp to soften length.
	dist = max(dist, SOFTENING);
	distSqrt = max(distSqrt, SOFTENING_SQRT);

	float force = massA * massB / dist;
	float2 ret = (dir * force) / distSqrt;

	return ret;
}


__kernel void CalcForce(global Body* inBodies, int inBodiesLen, global int2* meshIdxs, global int2* meshNBounds, global int2* meshBodyBounds, global int2* meshChildBounds, global float4* meshCMM, global int4* meshSPL, global int* meshNeighbors, const SimSettings sim, const SPHPreCalc sph, int meshTopStart, int meshTopEnd, global int* postNeeded)
{
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy body pos & mass.
	float2 iPos = (float2)(inBodies[(a)].PosX, inBodies[(a)].PosY);
	float iMass = inBodies[(a)].Mass;
	float iSize = inBodies[(a)].Size;

	// Copy body mesh cell.
	int bodyMeshId = inBodies[(a)].MeshID;
	int bodyCellParentID = meshSPL[bodyMeshId].y;
	int2 bodyCellIdx = meshIdxs[bodyMeshId];

	float2 iForce = (float2)(0.0f, 0.0f);
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
		int2 bodyCellParentNB = meshNBounds[bodyCellParentID];

		// Iterate parent cell neighbors.
		int start = bodyCellParentNB.x;
		int len = start + bodyCellParentNB.y;
		for (int nc = start; nc < len; nc++)
		{
			// Iterate neighbor child cells.
			int nId = meshNeighbors[(nc)];
			int childStartIdx = meshChildBounds[nId].x;
			int childLen = childStartIdx + meshChildBounds[nId].y;
			for (int c = childStartIdx; c < childLen; c++)
			{
				int2 childIdx = meshIdxs[c];

				// If the cell is far, compute force from cell.
				// [ Particle -> Mesh ]
				bool far = IsFar(bodyCellIdx, childIdx);
				if (far)
				{
					float4 cellCMM = meshCMM[c];
					iForce += CellForce(cellCMM.xy, iPos, cellCMM.z, iMass);
				}
				else if (bottom && !far) // Otherwise compute force from cell bodies if we are on the bottom level. [ Particle -> Particle ]
				{
					// Iterate the bodies within the cell.
					int2 cellBodyBounds = meshBodyBounds[c];
					int mbStart = cellBodyBounds.x;
					int mbLen = cellBodyBounds.y + mbStart;
					for (int mb = mbStart; mb < mbLen; mb++)
					{
						// Save us from ourselves.
						if (mb != a)
						{
							float2 jPos = (float2)(inBodies[(mb)].PosX, inBodies[(mb)].PosY);
							float jMass = inBodies[(mb)].Mass;
							float2 dir = jPos - iPos;
							float dist = dot(dir, dir);
							float distSqrt = SQRT(dist);

							// If this body is within collision/SPH distance.
							// [ SPH ]
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
							float force = jMass * iMass / dist;
							iForce += (dir * force) / distSqrt;
						}
					}
				}
			}
		}

		// Move to next parent level.
		bodyCellIdx = meshIdxs[bodyCellParentID];
		bodyCellParentID = meshSPL[bodyCellParentID].y;

		done = (bodyCellParentID == -1);
		bottom = false;
	}

	// *** Particle 2 Mesh ***
	// Accumulate force from remaining distant cells at the top-most level.
	for (int top = meshTopStart; top < meshTopEnd; top++)
	{
		int2 topIdx = meshIdxs[top];

		if (IsFar(bodyCellIdx, topIdx))
		{
			float4 topCMM = meshCMM[top];
			iForce += CellForce(topCMM.xy, iPos, topCMM.z, iMass);
		}
	}

	// Calculate pressure from density.
	iPressure = sim.GasK * iDensity;

	// Check for the phony roche condition.
	if (fabs(fast_length(iForce)) > (iMass * 4.0f) || iSize <= 1.1f)
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
	inBodies[(a)].Density = iDensity;
	inBodies[(a)].Pressure = iPressure;
}


// Is the specified cell a neighbor of the test cell?
bool IsFar(int2 cell, int2 testCell)
{
	uint2 diff = abs_diff(cell, testCell);
	if (diff.x > 1 || diff.y > 1)
		return true;

	return false;
}


__kernel void ElasticCollisions(global Body* inBodies, int inBodiesLen, global int4* meshSPL, global int2* meshNBounds, global int2* meshBodyBounds, global int* meshNeighbors, int collisions, global int* postNeeded)
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
		int4 parentSPL = meshSPL[outBody.MeshID];
		int pcellSize = parentSPL.x;
		int parentID = parentSPL.y;

		// Move up through parent cells until we find one
		// whose size is atleast as big as the target body.
		while (pcellSize < outBody.Size)
		{
			// Stop if we reach the top-most level.
			if (parentID == -1)
				break;

			parentID = parentSPL.y;
			parentSPL = meshSPL[parentID];
			pcellSize = parentSPL.x;
		}

		// Itereate the neighboring cells of the selected parent.
		int2 parentNBounds = meshNBounds[parentID];

		for (int i = parentNBounds.x; i < parentNBounds.x + parentNBounds.y; i++)
		{
			// Get the neighbor cell from the index.
			int nId = meshNeighbors[i];
			int2 nCellBodyBounds = meshBodyBounds[nId];

			// Iterate all the bodies within each neighboring cell.
			int mbStart = nCellBodyBounds.x;
			int mbLen = nCellBodyBounds.y + mbStart;
			for (int mb = mbStart; mb < mbLen; mb++)
			{
				// Save us from ourselves.
				if (mb != a)
				{
					Body inBody = inBodies[mb];

					// Calc the distance and check for collision.
					float distX = outBody.PosX - inBody.PosX;
					float distY = outBody.PosY - inBody.PosY;

					float dist = distX * distX + distY * distY;
					float distSqrt = (float)SQRT(dist);

					float colDist = outBody.Size * 0.5f + inBody.Size * 0.5f;
					if (distSqrt <= colDist)
					{
						// Calculate elastic collision forces.
						float colScale = (distX * (inBody.VeloX - outBody.VeloX) + distY * (inBody.VeloY - outBody.VeloY)) / dist;
						float forceX = distX * colScale;
						float forceY = distY * colScale;
						float colMass = inBody.Mass / (inBody.Mass + outBody.Mass);

						// If we're the bigger one, eat the other guy.
						if (outBody.Mass > inBody.Mass)
						{
							outBody = CollideBodies(outBody, inBody, colMass, forceX, forceY);
							inBodies[a] = outBody;
							inBodies[mb] = SetFlagB(inBodies[mb], CULLED, true);
							postNeeded[0] = 1;

						}
						else if (outBody.Mass == inBody.Mass) // If we are the same size, use a different metric.
						{
							// Our UID is more gooder, eat the other guy.
							if (outBody.UID > inBody.UID)
							{
								outBody = CollideBodies(outBody, inBody, colMass, forceX, forceY);
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


__kernel void SPHCollisions(global Body* inBodies, int inBodiesLen, global Body* outBodies, global int2* meshNBounds, global int4* meshSPL, global int2* meshChildBounds, global int2* meshIdxs, global int2* meshBodyBounds, global int* meshNeighbors, global float2* centerMass, const SimSettings sim, const SPHPreCalc sph, global int* postNeeded)
{
	// Get index for the current body.
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body from memory.
	Body outBody = inBodies[(a)];
	float2 outPos = (float2)(outBody.PosX, outBody.PosY);

	int bodyCellParentID = meshSPL[outBody.MeshID].y;
	int2 bodyCellIdx = meshIdxs[outBody.MeshID];

	if (sim.CollisionsOn == 1 && HasFlagB(outBody, INROCHE) && !HasFlagB(outBody, BLACKHOLE))
	{
		// Iterate parent cell neighbors.
		int2 parentNBounds = meshNBounds[bodyCellParentID];

		int start = parentNBounds.x;
		int len = start + parentNBounds.y;

		// PERF HACK: Mask out the len for bodies at resting density to skip the tree walk?
		len = len * !((outBody.Mass * sph.fDensity) == outBody.Density);

		for (int nc = start; nc < len; nc++)
		{
			// Iterate neighbor child cells.
			int nId = meshNeighbors[(nc)];

			int2 childBounds = meshChildBounds[nId];
			int childStartIdx = childBounds.x;
			int childLen = childStartIdx + childBounds.y;

			for (int c = childStartIdx; c < childLen; c++)
			{
				int2 cellIdx = meshIdxs[c];

				// Check for close cell.
				if (!IsFar(bodyCellIdx, cellIdx))
				{
					int2 cellBodyBounds = meshBodyBounds[c];
					int mbStart = cellBodyBounds.x;
					int mbLen = mbStart + cellBodyBounds.y;

					// Iterate the neighbor cell bodies.
					for (int mb = mbStart; mb < mbLen; mb++)
					{
						// Double tests are bad.
						if (mb != a)
						{
							Body inBody = inBodies[(mb)];
							float2 inPos = (float2)(inBody.PosX, inBody.PosY);
							float2 dir = outPos - inPos;
							float dist = dot(dir, dir);
							float distSqrt = SQRT(dist);

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

								// Viscosity force
								float viscLaplace = sph.fViscosity * kDiff;
								float viscScalar = inBody.Mass * viscLaplace * sim.Viscosity / inBody.Density;

								float veloDiffX = inBody.VeloX - outBody.VeloX;
								float veloDiffY = inBody.VeloY - outBody.VeloY;

								outBody.ForceX += veloDiffX * viscScalar;
								outBody.ForceY += veloDiffY * viscScalar;
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

	outBody.PosX += sim.DeltaTime * outBody.VeloX;
	outBody.PosY += sim.DeltaTime * outBody.VeloY;

	if (outBody.Lifetime > 0.0f)
	{
		outBody.Lifetime -= sim.DeltaTime * 4.0f;
	}

	// Check for and cull NANs.
	int nanCheck = isnan(outBody.PosX) + isnan(outBody.PosY);
	if (nanCheck > 0)
	{
		outBody.Flag = SetFlag(outBody.Flag, CULLED, true);
		postNeeded[0] = 1;
	}

	// Cull distant bodies.
	float2 cm = centerMass[0];
	float dist = DISTANCE(cm, (float2)(outBody.PosX, outBody.PosY));

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

Body CollideBodies(Body bodyA, Body bodyB, float colMass, float forceX, float forceY)
{
	Body outBody = bodyA;

	/*outBody.VeloX += colMass * forceX;
	outBody.VeloY += colMass * forceY;*/

	// Don't increase size of black holes.
	if (!HasFlagB(outBody, BLACKHOLE))
	{
		outBody.VeloX += colMass * forceX;
		outBody.VeloY += colMass * forceY;

		float a1 = pow((outBody.Size * 0.5f), 2.0f);
		float a2 = pow((bodyB.Size * 0.5f), 2.0f);
		float area = a1 + a2;
		outBody.Size = (float)native_sqrt(area) * 2.0f;
	}

	outBody.Mass += bodyB.Mass;

	return outBody;
}


//
// *** SORTING KERNELS ***
// Credit: https://github.com/gyatskov/radix-sort
// https://github.com/modelflat/OCLRadixSort
//

#define DataType long2
#define _BITS 8
#define _RADIX 256

// compute the histogram for each radix and each virtual processor for the pass
__kernel void histogram(const __global DataType* restrict d_Keys, __global int* restrict d_Histograms, const int pass, __local int* loc_histo, const int n)
{
	int it = get_local_id(0);  // i local number of the processor
	int ig = get_global_id(0); // global number = i + g I
	int gr = get_group_id(0); // gr group number
	const int groups = get_num_groups(0);
	int items = get_local_size(0);

	// initialize the local histograms to zero
	for (int ir = 0; ir < _RADIX; ir++)
	{
		loc_histo[ir * items + it] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// range of keys that are analyzed by the work item
	int sublist_size = n / groups / items; // size of the sub-list
	int sublist_start = ig * sublist_size; // beginning of the sub-list

	long key;
	long shortkey;
	int k;

	// compute the index
	// the computation depends on the transposition
	for (int j = 0; j < sublist_size; j++)
	{
		k = j + sublist_start;

		key = d_Keys[k].x;

		// extract the group of _BITS bits of the pass
		// the result is in the range 0.._RADIX-1
		// _BITS = size of _RADIX in bits. So basically they
		// represent both the same. 
		shortkey = ((key >> (pass * _BITS)) & (_RADIX - 1)); // _RADIX-1 to get #_BITS "ones"

		// increment the local histogram
		loc_histo[shortkey * items + it]++;
	}

	// wait for local histogram to finish
	barrier(CLK_LOCAL_MEM_FENCE);

	// copy the local histogram to the global one
	// in this case the global histo is the group histo.
	for (int ir = 0; ir < _RADIX; ir++)
	{
		d_Histograms[items * (ir * groups + gr) + it] = loc_histo[ir * items + it];
	}
}

// perform a parallel prefix sum (a scan) on the local histograms
// (see Blelloch 1990) each workitem worries about two memories
// see also http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
__kernel void scanhistograms(__global int* histo, __local int* temp, __global int* globsum)
{
	int it = get_local_id(0);
	int ig = get_global_id(0);
	int decale = 1;
	int n = get_local_size(0) << 1;
	int gr = get_group_id(0);

	// load input into local memory
	// up sweep phase
	temp[(it << 1)] = histo[(ig << 1)];
	temp[(it << 1) + 1] = histo[(ig << 1) + 1];

	// parallel prefix sum (algorithm of Blelloch 1990)
	// This loop runs log2(n) times
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);

		if (it < d)
		{
			int ai = decale * ((it << 1) + 1) - 1;
			int bi = decale * ((it << 1) + 2) - 1;
			temp[bi] += temp[ai];
		}

		decale <<= 1;
	}

	// store the last element in the global sum vector
	// (maybe used in the next step for constructing the global scan)
	// clear the last element
	if (it == 0)
	{
		globsum[gr] = temp[n - 1];
		temp[n - 1] = 0;
	}

	// down sweep phase
	// This loop runs log2(n) times
	for (int d = 1; d < n; d <<= 1)
	{
		decale >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if (it < d)
		{
			int ai = decale * ((it << 1) + 1) - 1;
			int bi = decale * ((it << 1) + 2) - 1;

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// write results to device memory
	histo[(ig << 1)] = temp[(it << 1)];
	histo[(ig << 1) + 1] = temp[(it << 1) + 1];
}

// use the global sum for updating the local histograms
// each work item updates two values
__kernel void pastehistograms(__global int* restrict histo, const __global int* restrict globsum)
{
	int ig = get_global_id(0);
	int gr = get_group_id(0);

	int s = globsum[gr];

	// write results to device memory
	histo[(ig << 1)] += s;
	histo[(ig << 1) + 1] += s;
}

// each virtual processor reorders its data using the scanned histogram
__kernel void reorder(const __global DataType* restrict d_inKeys, __global DataType* restrict d_outKeys, const __global int* d_Histograms, const int pass, __local  int* loc_histo, const int n)
{
	int it = get_local_id(0);  // i local number of the processor
	int ig = get_global_id(0); // global number = i + g I
	int gr = get_group_id(0);				// gr group number
	const int groups = get_num_groups(0);	// G: group count
	int items = get_local_size(0);			// group size

	int start = ig * (n / groups / items);   // index of first elem this work-item processes
	int size = n / groups / items;			// count of elements this work-item processes

	// take the histogram in the cache
	for (int ir = 0; ir < _RADIX; ir++)
	{
		loc_histo[ir * items + it] = d_Histograms[items * (ir * groups + gr) + it];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int newpos;					// new position of element
	long2 key;		// key element
	long shortkey;	// key element within cache (cache line)
	int k;						// global position within input elements

	for (int j = 0; j < size; j++)
	{
		k = j + start;
		key = d_inKeys[k];
		shortkey = ((key.x >> (pass * _BITS)) & (_RADIX - 1));	// shift element to relevant bit positions

		newpos = loc_histo[shortkey * items + it];

		d_outKeys[newpos] = key;

		newpos++;
		loc_histo[shortkey * items + it] = newpos;
	}
}