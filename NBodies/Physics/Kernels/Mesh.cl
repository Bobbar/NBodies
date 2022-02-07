
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


// Computes the morton numbers from body positions.
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

// Read indexes from the sorted morton buffer and copy bodies to their sorted location.
__kernel void ReindexBodies(global Body* inBodies, int blen, global long2* sortMap, global Body* outBodies)
{
	int b = get_global_id(0);

	if (b >= blen)
		return;

	int newIdx = (int)sortMap[b].y;

	outBodies[b] = inBodies[newIdx];
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
