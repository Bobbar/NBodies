

typedef struct __attribute__((packed)) Body
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
	float PosZ;
	float VeloZ;
	float ForceZ;
	float Lifetime;
	int MeshID;

} Body;


typedef struct __attribute__((packed)) MeshCell
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
	int GridIdx;

} MeshCell;


typedef struct __attribute__((packed)) GridInfo
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
	int Columns;
	int Rows;
	int Layers;
	int Size;
	int IndexOffset;

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

float3 ComputeForce(float3 posA, float3 posB, float massA, float massB);
int GridHash(int column, int row, int layer, GridInfo grid);
bool IsNeighbor(MeshCell cell, MeshCell testCell);
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


__kernel void ReindexBodies(global Body* inBodies, int blen, global int* sortMap, global Body* outBodies)
{
	int b = get_global_id(0);

	if (b >= blen)
		return;

	/*if (inBodies[sortMap[b]].PosY > 0.0f)
	{
		printf("%i\n", b);
	}*/

	outBodies[b] = inBodies[sortMap[b]];
}


__kernel void ClearGrid(global int* gridIdx, int passStride, int passOffset, global MeshCell* mesh, int meshLen)
{
	int m = get_global_id(0);

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

__kernel void PopGrid(global int* gridIdx, int passStride, int passOffset, global GridInfo* gridInfo, global MeshCell* mesh, int meshLen)
{
	int m = get_global_id(0);

	if (m >= meshLen)
		return;

	MeshCell cell;
	cell.IdxX = mesh[m].IdxX;
	cell.IdxY = mesh[m].IdxY;
	cell.IdxZ = mesh[m].IdxZ;

	cell.Level = mesh[m].Level;
	cell.GridIdx = 0;

	GridInfo grid;
	grid.OffsetX = gridInfo[cell.Level].OffsetX;
	grid.OffsetY = gridInfo[cell.Level].OffsetY;
	grid.OffsetZ = gridInfo[cell.Level].OffsetZ;

	grid.Rows = gridInfo[cell.Level].Rows;
	grid.Columns = gridInfo[cell.Level].Columns;
	grid.IndexOffset = gridInfo[cell.Level].IndexOffset;

	// Compute bucket index.
	int column = cell.IdxX + grid.OffsetX;
	int row = cell.IdxY + grid.OffsetY;
	int layer = cell.IdxZ + grid.OffsetZ;

	//int bucket = (row * grid.Columns) + column + row;
	int bucket = GridHash(column, row, layer, grid);//((layer * grid.Rows) * grid.Columns) + (row * gird.Columns) + column;


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
		gridIdx[bucket] = m;
	}

	mesh[m].GridIdx = cell.GridIdx;
}

int GridHash(int column, int row, int layer, GridInfo grid)
{
	return ((layer * grid.Rows) * grid.Columns) + (row * grid.Columns) + column;
}

__kernel void BuildNeighbors(global MeshCell* mesh, int meshLen, global GridInfo* gridInfo, global int* gridIdx, int passStride, int passOffset, global int* neighborIndex)
{
	int m = get_global_id(0);

	if (m >= meshLen)
		return;

	int offset = m * 27;

	int count = 0;
	int cellLevel = mesh[m].Level;
	int cellGridIdx = mesh[m].GridIdx;

	//	int3 shiftLut[] = { { -1,-1,-1 },{ 0,-1,-1 },{ 1,-1,-1 },{ -1,0,-1 },{ 0,0,-1 },{ 1,0,-1 },{ -1,1,-1 },{ 0,1,-1 },{ 1,1,-1 },{ -1,-1,0 },{ 0,-1,0 },{ 1,-1,0 },{ -1,0,0 },{ 0,0,0 },{ 1,0,0 },{ -1,1,0 },{ 0,1,0 },{ 1,1,0 },{ -1,-1,1 },{ 0,-1,1 },{ 1,-1,1 },{ -1,0,1 },{ 0,0,1 },{ 1,0,1 },{ -1,1,1 },{ 0,1,1 },{ 1,1,1 } };

	GridInfo grid = gridInfo[cellLevel];

	//int gColumns = grid.Columns;
	int gIndexOffset = grid.IndexOffset;
	int gSize = grid.Size;

	//int gColumns = gridInfo[cellLevel].Columns;
	//int gIndexOffset = gridInfo[cellLevel].IndexOffset;
	//int gSize = gridInfo[cellLevel].Size;
	int offsetIndex = cellGridIdx - passOffset;

	if (offsetIndex >= 0 && offsetIndex < passStride)
	{
		int offIdx = cellGridIdx - gIndexOffset;

		// Shift bucket index around the cell and check for populated grid index buckets.

		for (int x = -1; x <= 1; x++)
		{
			for (int y = -1; y <= 1; y++)
			{
				for (int z = -1; z <= 1; z++)
				{
					int localIdx = offIdx + GridHash(x, y, z, grid);  // Does this work with the new hash algo?

					if (localIdx > 0 && localIdx < gSize)
					{
						int bucket = localIdx + gIndexOffset;

						bucket -= passOffset;

						if (bucket >= 0 && bucket < passStride)
						{
							int idx = gridIdx[bucket];

							// Check for populated bucket and poplate neighbor index.
							if (idx >= 0)
							{
								neighborIndex[(offset + count)] = idx;
								count++;
							}
						}
					}
				}
			}
		}

		//for (int i = 0; i < 27; i++)
		//{
		//	int3 shift = shiftLut[i];
		//	//int localIdx = offIdx + ((shift.y * gColumns) + (shift.x + shift.y));
		//	int localIdx = offIdx + GridHash(shift.x, shift.y, shift.z, grid);

		//	if (localIdx > 0 && localIdx < gSize)
		//	{
		//		int bucket = localIdx + gIndexOffset;

		//		bucket -= passOffset;

		//		if (bucket >= 0 && bucket < passStride)
		//		{
		//			int idx = gridIdx[bucket];

		//			// Check for populated bucket and poplate neighbor index.
		//			if (idx >= 0)
		//			{
		//				neighborIndex[(offset + count)] = idx;
		//				count++;
		//			}
		//		}
		//	}
		//}

		//// Add this cell to end.
		//neighborIndex[(offset + count++)] = m;

		// Set cell neighbor index list pointers.
		mesh[m].NeighborStartIdx = offset;
		mesh[m].NeighborCount = count;
	}
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
	float distX = posA.x - posB.x;
	float distY = posA.y - posB.y;
	float distZ = posA.z - posB.z;

	float dist = distX * distX + distY * distY + distZ * distZ;
	float distSqrt = (float)native_sqrt(dist);
	dist = max(dist, SOFTENING);
	distSqrt = max(distSqrt, SOFTENING_SQRT);
	float force = massA * massB / dist;

	float3 ret;
	ret.x = force * distX / distSqrt;
	ret.y = force * distY / distSqrt;
	ret.z = force * distZ / distSqrt;

	return ret;
}

__kernel void CalcForce(global Body* inBodies, int inBodiesLen, global MeshCell* inMesh, int meshTopStart, int meshTopEnd, global int* meshNeighbors, const SimSettings sim, const SPHPreCalc sph, global int* postNeeded)
{
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body and mesh cell from memory.
	MeshCell levelCell = inMesh[(inBodies[(a)].MeshID)];
	MeshCell levelCellParent = levelCell;

	float3 bPos = (float3)(inBodies[(a)].PosX, inBodies[(a)].PosY, inBodies[(a)].PosZ);
	float bMass = inBodies[(a)].Mass;
	int bFlags = inBodies[(a)].Flag;
	float3 bForce = (float3)(0.0f, 0.0f, 0.0f);
	float bDensity = 0.0f;
	float bPressure = 0.0f;
	float totForce = 0;

	// Resting Density.	
	bDensity = bMass * sph.fDensity;

	// *** Particle 2 Particle & SPH ***
	// Accumulate forces from all bodies within neighboring cells. [THIS INCLUDES THE BODY'S OWN CELL]
	for (int n = levelCell.NeighborStartIdx; n < levelCell.NeighborStartIdx + levelCell.NeighborCount; n++)
	{
		// Get the mesh cell index, then copy it from memory.
		int nId = meshNeighbors[(n)];
		MeshCell cell = inMesh[(nId)];

		// Iterate the bodies within the cell.
		// Read from body array at the correct location.
		int mbStart = cell.BodyStartIdx;
		int mbLen = cell.BodyCount + mbStart;
		for (int mb = mbStart; mb < mbLen; mb++)
		{
			// Save us from ourselves.
			if (mb != a)
			{
				Body inBody = inBodies[(mb)];

				float distX = inBody.PosX - bPos.x;
				float distY = inBody.PosY - bPos.y;
				float distZ = inBody.PosZ - bPos.z;

				float dist = distX * distX + distY * distY + distZ * distZ;
				float distSqrt = (float)native_sqrt(dist);

				// If this body is within collision/SPH distance.
				if (distSqrt <= sph.kSize)
				{
					// Clamp SPH softening distance.
					dist = max(dist, SPH_SOFTENING);

					// Accumulate bDensity.
					float diff = sph.kSizeSq - dist;
					float fac = sph.fDensity * diff * diff * diff;
					bDensity += bMass * fac;
				}

				// Clamp gravity softening distance.
				dist = max(dist, SOFTENING);
				distSqrt = max(distSqrt, SOFTENING_SQRT);

				// Accumulate body-to-body force.
				float force = inBody.Mass * bMass / dist;

				totForce += force;
				bForce.x += force * distX / distSqrt;
				bForce.y += force * distY / distSqrt;
				bForce.z += force * distZ / distSqrt;

			}
		}
	}

	// *** Particle 2 Mesh ***
	// Accumulate force from neighboring cells at each level.
	while (levelCellParent.ParentID != -1)
	{
		levelCellParent = inMesh[(levelCellParent.ParentID)];

		// Iterate parent cell neighbors, skipping the last neighbor which is the parent.
		int start = levelCellParent.NeighborStartIdx;
		int len = start + levelCellParent.NeighborCount;  // - 1;

		for (int nc = start; nc < len; nc++)
		{
			int nId = meshNeighbors[(nc)];
			MeshCell nCell = inMesh[(nId)];

			// Iterate neighbor child cells.
			int childStartIdx = nCell.ChildStartIdx;
			int childLen = childStartIdx + nCell.ChildCount;
			for (int c = childStartIdx; c < childLen; c++)
			{
				MeshCell cell = inMesh[(c)];

				if (!IsNeighbor(levelCell, cell))
				{
					float3 cellPos = (float3)(cell.CmX, cell.CmY, cell.CmZ);
					bForce += ComputeForce(cellPos, bPos, cell.Mass, bMass);

					//// Calculate the force from the cells center of mass.
					//float distX = cell.CmX - bPos.x;
					//float distY = cell.CmY - bPos.y;
					//float dist = distX * distX + distY * distY;
					//float distSqrt = (float)native_sqrt(dist);
					//float force = cell.Mass * bMass / dist;

					//totForce += force;
					//bForce.x += force * distX / distSqrt;
					//bForce.y += force * distY / distSqrt;
				}
			}
		}

		levelCell = levelCellParent;
	}

	// *** Particle 2 Mesh ***
	// Accumulate force from remaining distant cells at the top-most level.
	for (int top = meshTopStart; top < meshTopEnd; top++)
	{
		MeshCell cell = inMesh[(top)];

		if (!IsNeighbor(levelCell, cell))
		{

			float3 cellPos = (float3)(cell.CmX, cell.CmY, cell.CmZ);
			bForce += ComputeForce(cellPos, bPos, cell.Mass, bMass);

			/*	float distX = cell.CmX - bPos.x;
				float distY = cell.CmY - bPos.y;
				float dist = distX * distX + distY * distY;
				float distSqrt = (float)native_sqrt(dist);
				float force = cell.Mass * bMass / dist;

				totForce += force;
				bForce.x += force * distX / distSqrt;
				bForce.y += force * distY / distSqrt;*/
		}
	}

	// Calculate pressure from density.
	bPressure = sim.GasK * bDensity;

	if (totForce > bMass * 4.0f)
	{
		int newFlags = SetFlag(bFlags, INROCHE, true);
		if (newFlags != bFlags)
		{
			bFlags = newFlags;
			postNeeded[0] = 1;
		}
	}

	// Write back to memory.
	inBodies[(a)].ForceX = bForce.x;
	inBodies[(a)].ForceY = bForce.y;
	inBodies[(a)].ForceZ = bForce.z;

	inBodies[(a)].Density = bDensity;
	inBodies[(a)].Pressure = bPressure;
	inBodies[(a)].Flag = bFlags;
}

// Is the specified cell a neighbor of the test cell?
bool IsNeighbor(MeshCell cell, MeshCell testCell)
{
	/*if (testCell.IdxX > cell.IdxX + -2 && testCell.IdxX < cell.IdxX + 2 && testCell.IdxY > cell.IdxY + -2 && testCell.IdxY < cell.IdxY + 2 && testCell.IdxZ > cell.IdxZ + -2 && testCell.IdxZ < cell.IdxZ + 2)
	{
		return true;
	}
	return false;*/


	if (abs(cell.IdxX - testCell.IdxX) > 1 || abs(cell.IdxY - testCell.IdxY) > 1 || abs(cell.IdxZ - testCell.IdxZ) > 1)
	{
		return false;
	}

	return true;
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
					float distSqrt = (float)native_sqrt(dist);

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

	// Copy this body's mesh cell from memory.
	MeshCell bodyCell = inMesh[(outBody.MeshID)];

	if (sim.CollisionsOn == 1)
	{
		// Iterate neighbor cells.
		for (int i = bodyCell.NeighborStartIdx; i < bodyCell.NeighborStartIdx + bodyCell.NeighborCount; i++)
		{
			// Get the neighbor cell from the index.
			int nId = meshNeighbors[(i)];
			MeshCell cell = inMesh[(nId)];

			// Iterate the neighbor cell bodies.
			int mbStart = cell.BodyStartIdx;
			int mbLen = cell.BodyCount + mbStart;
			for (int mb = mbStart; mb < mbLen; mb++)
			{
				// Double tests are bad.
				if (mb != a)
				{
					Body inBody = inBodies[(mb)];

					float distX = outBody.PosX - inBody.PosX;
					float distY = outBody.PosY - inBody.PosY;
					float distZ = outBody.PosZ - inBody.PosZ;

					float dist = distX * distX + distY * distY + distZ * distZ;
					float distSqrt = (float)native_sqrt(dist);
					// Calc the distance and check for collision.
					//float colDist = (sph.kSize * 0.5f) * 2.0f;
					//if (dist <= sph.kSize * sph.kSize)
					if (distSqrt <= sph.kSize)
					{
						//// Handle exact overlaps.
						//if (dist == 0)
						//{
						//	outBody.PosX += (outBody.UID + 1) * SPH_SOFTENING;
						//	outBody.PosY += (outBody.UID + 1) * SPH_SOFTENING;
						//}

						dist = max(dist, SPH_SOFTENING);

						// Only do SPH collision if both bodies are in roche.
						// SPH collision.
						if (HasFlagB(outBody, INROCHE) && HasFlagB(inBody, INROCHE))
						{
							//float distSqrt = (float)native_sqrt(dist);
							distSqrt = max(distSqrt, SPH_SOFTENING);

							float kDiff = sph.kSize - distSqrt;

							// Pressure force
							float pressScalar = outBody.Mass * (outBody.Pressure + inBody.Pressure) / (2.0f * outBody.Density);
							float pressGrad = sph.fPressure * kDiff * kDiff / distSqrt;

							outBody.ForceX += (distX * pressGrad) * pressScalar;
							outBody.ForceY += (distY * pressGrad) * pressScalar;
							outBody.ForceZ += (distZ * pressGrad) * pressScalar;

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
	float dist = fast_distance(cm, (float3)(outBody.PosX, outBody.PosY, outBody.PosZ));

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