
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
__kernel void CompressMap(int blocks, global int* cellmapIn, global int* cellmapOut, global int* counts, global int* levelCounts, global int* levelIdx, int level, int threads)
{
	int gid = get_global_id(0);
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

	// Make sure we don't hit a padded element.
	// This condition may be safe to remove.
	//if (newIdx > -1)
	outBodies[b] = inBodies[newIdx];
}


//__kernel void BuildBottom(global Body* inBodies, global int4* meshIdxs, global int2* meshBodyBounds, global float4* meshCMM, global int2* meshSPL, global int* levelCounts, int bodyLen, global int* cellMap, int cellSizeExp, int cellSize, global long* parentMorts, long bufLen)
//{
//	int m = get_global_id(0);
//
//	int meshLen = levelCounts[0];
//
//	if (m >= meshLen || m >= bufLen)
//		return;
//
//	int firstIdx = 0;
//	if (m > 0)
//		firstIdx = cellMap[m - 1];
//
//	int lastIdx = cellMap[m];
//	if (m == meshLen - 1)
//		lastIdx = bodyLen;
//
//	float fPosX = inBodies[firstIdx].PosX;
//	float fPosY = inBodies[firstIdx].PosY;
//	float fPosZ = inBodies[firstIdx].PosZ;
//	float fMass = inBodies[firstIdx].Mass;
//
//	double3 nCM = (double3)(fMass * fPosX, fMass * fPosY, fMass * fPosZ);
//	double nMass = fMass;
//
//	int4 meshIdx = (int4)((int)floor(fPosX) >> cellSizeExp, (int)floor(fPosY) >> cellSizeExp, (int)floor(fPosZ) >> cellSizeExp, 0);
//	meshIdxs[m] = meshIdx;
//	meshBodyBounds[m] = (int2)(firstIdx, lastIdx - firstIdx);
//	meshSPL[m] = (int2)(cellSize, -1);
//
//	// Compute parent level morton numbers.
//	int idxX = meshIdx.x >> 1;
//	int idxY = meshIdx.y >> 1;
//	int idxZ = meshIdx.z >> 1;
//	long morton = MortonNumber(idxX, idxY, idxZ);
//	parentMorts[m] = morton;
//
//	inBodies[firstIdx].MeshID = m;
//
//	for (int i = firstIdx + 1; i < lastIdx; i++)
//	{
//		float posX = inBodies[i].PosX;
//		float posY = inBodies[i].PosY;
//		float posZ = inBodies[i].PosZ;
//		float mass = inBodies[i].Mass;
//
//		nMass += mass;
//		nCM.x += mass * posX;
//		nCM.y += mass * posY;
//		nCM.z += mass * posZ;
//
//		inBodies[i].MeshID = m;
//	}
//
//	meshCMM[m] = (float4)((nCM.x / nMass), (nCM.y / nMass), (nCM.z / nMass), nMass);
//}
__kernel void BuildBottom(global Body* inBodies, global Body* outBodies, global long2* sortMap, global int4* meshIdxs, global int2* meshBodyBounds, global float4* meshCMM, global int2* meshSPL, global int* levelCounts, int bodyLen, global int* cellMap, int cellSizeExp, int cellSize, global long* parentMorts, long bufLen)
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

	// Get the first body from the unsorted location.
	Body firstBod = inBodies[sortMap[firstIdx].y];
	float fPosX = firstBod.PosX;
	float fPosY = firstBod.PosY;
	float fPosZ = firstBod.PosZ;
	float fMass = firstBod.Mass;

	// Set the mesh ID.
	firstBod.MeshID = m;

	double3 nCM = (double3)(fMass * fPosX, fMass * fPosY, fMass * fPosZ);
	double nMass = fMass;

	int4 meshIdx = (int4)((int)floor(fPosX) >> cellSizeExp, (int)floor(fPosY) >> cellSizeExp, (int)floor(fPosZ) >> cellSizeExp, 0);
	meshIdxs[m] = meshIdx;
	meshBodyBounds[m] = (int2)(firstIdx, lastIdx - firstIdx);
	meshSPL[m] = (int2)(cellSize, -1);

	// Compute parent level morton numbers.
	int idxX = meshIdx.x >> 1;
	int idxY = meshIdx.y >> 1;
	int idxZ = meshIdx.z >> 1;
	long morton = MortonNumber(idxX, idxY, idxZ);
	parentMorts[m] = morton;

	// Copy the body to its sorted location.
	outBodies[firstIdx] = firstBod;

	for (int i = firstIdx + 1; i < lastIdx; i++)
	{
		// Get the child body from the unsorted location.
		Body childBod = inBodies[sortMap[i].y];

		float posX = childBod.PosX;
		float posY = childBod.PosY;
		float posZ = childBod.PosZ;
		float mass = childBod.Mass;

		nMass += mass;
		nCM.x += mass * posX;
		nCM.y += mass * posY;
		nCM.z += mass * posZ;

		childBod.MeshID = m;

		// Copy the child body to its sorted location.
		outBodies[i] = childBod;
	}

	meshCMM[m] = (float4)((nCM.x / nMass), (nCM.y / nMass), (nCM.z / nMass), nMass);
}

__kernel void BuildTop(global int4* meshIdxs, global int2* meshBodyBounds, global int2* meshChildBounds, global float4* meshCMM, global int2* meshSPL, global int* levelCounts, global int* levelIdx, global int* cellMap, int cellSize, int level, global long* parentMorts, long bufLen)
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

	double3 nCM;
	double nMass;

	int4 firstMIdx = meshIdxs[firstIdx];
	float4 firstCMM = meshCMM[firstIdx];
	int2 bodyBounds = meshBodyBounds[firstIdx];

	int4 meshIdx = firstMIdx >> 1;
	int2 childBounds = (int2)(firstIdx, 1);

	nMass = (double)firstCMM.w;
	nCM = (double3)(nMass * firstCMM.x, nMass * firstCMM.y, nMass * firstCMM.z);

	// Compute parent level morton numbers.
	int idxX = meshIdx.x >> 1;
	int idxY = meshIdx.y >> 1;
	int idxZ = meshIdx.z >> 1;
	long morton = MortonNumber(idxX, idxY, idxZ);
	parentMorts[m] = morton;

	meshSPL[firstIdx].y = newIdx;

	for (int i = firstIdx + 1; i < lastIdx; i++)
	{
		bodyBounds.y += meshBodyBounds[i].y;
		childBounds.y++;

		float4 childCMM = meshCMM[i];
		float mass = childCMM.w;
		nMass += mass;
		nCM.x += mass * childCMM.x;
		nCM.y += mass * childCMM.y;
		nCM.z += mass * childCMM.z;

		meshSPL[i].y = newIdx;
	}

	meshIdxs[newIdx] = meshIdx;
	meshSPL[newIdx] = (int2)(cellSize, -1);
	meshChildBounds[newIdx] = childBounds;
	meshBodyBounds[newIdx] = bodyBounds;
	meshCMM[newIdx] = (float4)((nCM.x / nMass), (nCM.y / nMass), (nCM.z / nMass), nMass);
}


// Top-down mesh based nearest neighbor search.
__kernel void BuildNeighborsMesh(global int4* meshIdxs, global int2* meshSPL, global int2* meshNBounds, global int2* meshChildBounds, global int* neighborIndex, int botOffset, int levels, int level, int start, int end)
{
	int m = get_global_id(0);
	int readM = m + start;

	if (readM >= end)
		return;

	// Write location of the neighbor list.
	long offset = (readM - botOffset) * 27;
	int count = 0;

	int4 cellIdxs = meshIdxs[readM];
	int2 cellSPL = meshSPL[readM];

	if (level == levels)
	{
		// For the first pass, iterate all top-level cells and brute force the neighbors. (There won't be many, so this is fast.)
		for (int i = start; i < end; i++)
		{
			int4 checkIdxs = meshIdxs[i];

			if (!IsFar(cellIdxs, checkIdxs))
			{
				neighborIndex[(offset + count++)] = i;
			}
		}
	}
	else
	{
		// Use the mesh tree hierarchy & neighbors found for parent level to narrow down the search area significantly.
		int2 parentNBounds = meshNBounds[cellSPL.y];

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
				int4 childIdxs = meshIdxs[c];

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


__kernel void CalcCenterOfMass(global float4* meshCMM, global float3* cm, int start, int end)
{
	double cmX = 0;
	double cmY = 0;
	double cmZ = 0;
	double mass = 0;

	for (int i = start; i < end; i++)
	{
		float4 cellCMM = meshCMM[i];

		mass += cellCMM.w;
		cmX += cellCMM.w * cellCMM.x;
		cmY += cellCMM.w * cellCMM.y;
		cmZ += cellCMM.w * cellCMM.z;

	}

	cmX = cmX / mass;
	cmY = cmY / mass;
	cmZ = cmZ / mass;


	cm[0] = (float3)(cmX, cmY, cmZ);
}
