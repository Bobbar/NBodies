

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
	float Lifetime;
	int MeshID;

} Body;


typedef struct __attribute__((packed)) MeshCell
{
	int ID;
	int IdxX;
	int IdxY;
	float CmX;
	float CmY;
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
	long OffsetX;
	long OffsetY;
	long MinX;
	long MinY;
	long MaxX;
	long MaxY;
	long Columns;
	long Rows;
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

bool IsNeighbor(MeshCell cell, MeshCell testCell);
Body CollideBodies(Body master, Body slave, float colMass, float forceX, float forceY);
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
	int i = get_local_size(0) * get_group_id(0) + get_local_id(0);

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

	outBodies[b] = inBodies[sortMap[b]];
}


__kernel void ClearGrid(global int* gridIdx, int passStride, int passOffset, global MeshCell* mesh, int meshLen)
{
	int m = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (m >= meshLen)
	{
		return;
	}

	int idx = mesh[m].GridIdx;
	idx -= passOffset;

	if (idx >= 0 && idx < passStride)
	{
		gridIdx[idx] = 0;
	}

}

__kernel void PopGrid(global int* gridIdx, int passStride, int passOffset, global GridInfo* gridInfo, global MeshCell* mesh, int meshLen)
{
	int m = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (m >= meshLen)
	{
		return;
	}

	MeshCell cell = mesh[m];
	GridInfo grid = gridInfo[cell.Level];

	// Compute bucket index.
	int column = cell.IdxX + grid.OffsetX;
	int row = cell.IdxY + grid.OffsetY;
	int bucket = (row * grid.Columns) + column + row;

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
		// Unpopulated grid buckets = 0.
		// Cell ID of 0 == bucked index of -1.
		if (cell.ID == 0)
		{
			gridIdx[bucket] = -1;
		}
		else
		{
			gridIdx[bucket] = cell.ID;
		}
	}

	mesh[m] = cell;
}

__kernel void BuildNeighbors(global MeshCell* mesh, int meshLen, global GridInfo* gridInfo, global int* gridIdx, int passStride, int passOffset, global int* neighborIndex)
{
	int m = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (m >= meshLen)
	{
		return;
	}

	int offset = m * 9;
	int count = 0;
	MeshCell cell = mesh[m];
	GridInfo gInfo = gridInfo[cell.Level];
	int columns = gInfo.Columns;
	int offsetIndex = cell.GridIdx - passOffset;

	if (offsetIndex >= 0 && offsetIndex < passStride)
	{
		int offIdx = cell.GridIdx - gInfo.IndexOffset;

		// Shift bucket index around the cell and check for populated grid index buckets.
		for (int x = -1; x <= 1; x++)
		{
			for (int y = -1; y <= 1; y++)
			{
				int localIdx = offIdx + ((x * columns) + (y + x));

				if (localIdx > 0 && localIdx < gInfo.Size)
				{
					int bucket = localIdx + gInfo.IndexOffset;
					bucket -= passOffset;

					if (bucket >= 0 && bucket < passStride)
					{
						int idx = gridIdx[bucket];

						// Check for populated bucket and poplate neighbor index list accordingly.
						if (idx > 0)
						{
							neighborIndex[(offset + count)] = idx;
							count++;
						}
						else if (idx == -1)
						{
							neighborIndex[(offset + count)] = 0;
							count++;
						}
					}
				}
			}
		}

		// Set cell neighbor index list pointers.
		cell.NeighborStartIdx = offset;
		cell.NeighborCount = count;
	}

	mesh[m] = cell;
}

__kernel void BuildBottom(global Body* inBodies, global Body* outBodies, global MeshCell* mesh, int meshLen, global int* cellIdx, int cellSizeExp)
{
	int m = get_global_id(0);

	if (m >= meshLen)
		return;

	int firstIdx = cellIdx[m];
	Body firstBody = inBodies[firstIdx];

	MeshCell newCell;
	newCell.IdxX = (int)floor(firstBody.PosX) >> cellSizeExp;
	newCell.IdxY = (int)floor(firstBody.PosY) >> cellSizeExp;
	newCell.Size = (int)pown(2.0f, cellSizeExp);
	newCell.BodyStartIdx = firstIdx;
	newCell.BodyCount = 1;
	newCell.ChildCount = 0;
	newCell.ID = m;
	newCell.Level = 0;
	newCell.ParentID = -1;
	newCell.CmX = firstBody.Mass * firstBody.PosX;
	newCell.CmY = firstBody.Mass * firstBody.PosY;
	newCell.Mass = firstBody.Mass;

	firstBody.MeshID = m;
	outBodies[firstIdx] = firstBody;

	for (int i = firstIdx + 1; i < cellIdx[m + 1]; i++)
	{
		Body body = inBodies[i];
		newCell.Mass += body.Mass;
		newCell.CmX += body.Mass * body.PosX;
		newCell.CmY += body.Mass * body.PosY;
		newCell.BodyCount++;

		body.MeshID = m;
		outBodies[i] = body;
	}

	newCell.CmX = newCell.CmX / (float)newCell.Mass;
	newCell.CmY = newCell.CmY / (float)newCell.Mass;

	mesh[m] = newCell;
}

__kernel void BuildTop(global MeshCell* mesh, int len, global int* cellIdx, int cellSizeExp, int levelOffset, int meshOffset, int readOffset, int level)
{
	int m = get_global_id(0);

	if (m >= len)
		return;

	int locIdxOff = m + readOffset;
	int cellIdxOff = locIdxOff + (level - 1);
	int newIdx = m + meshOffset;

	int firstIdx = cellIdx[cellIdxOff] + levelOffset;
	MeshCell firstCell = mesh[firstIdx];

	MeshCell newCell;
	newCell.IdxX = firstCell.IdxX >> 1;
	newCell.IdxY = firstCell.IdxY >> 1;
	newCell.Size = (int)pown(2.0f, cellSizeExp);
	newCell.ChildStartIdx = firstIdx;
	newCell.ChildCount = 1;
	newCell.ID = newIdx;
	newCell.Level = level;
	newCell.BodyStartIdx = firstCell.BodyStartIdx;
	newCell.BodyCount = firstCell.BodyCount;
	newCell.ParentID = -1;
	newCell.CmX = (float)firstCell.Mass * firstCell.CmX;
	newCell.CmY = (float)firstCell.Mass * firstCell.CmY;
	newCell.Mass = firstCell.Mass;

	firstCell.ParentID = newIdx;
	mesh[firstIdx] = firstCell;

	for (int i = firstIdx + 1; i < cellIdx[cellIdxOff + 1] + levelOffset; i++)
	{
		MeshCell child = mesh[i];

		newCell.Mass += child.Mass;
		newCell.CmX += (float)child.Mass * child.CmX;
		newCell.CmY += (float)child.Mass * child.CmY;
		newCell.ChildCount++;
		newCell.BodyCount += child.BodyCount;

		child.ParentID = newIdx;
		mesh[i] = child;
	}

	newCell.CmX = newCell.CmX / (float)newCell.Mass;
	newCell.CmY = newCell.CmY / (float)newCell.Mass;

	mesh[newIdx] = newCell;
}

__kernel void CalcCenterOfMass(global MeshCell* inMesh, global float2* cm, int start, int end)
{
	double cmX = 0;
	double cmY = 0;
	double mass = 0;

	for (int i = start; i < end; i++)
	{
		MeshCell cell = inMesh[i];

		mass += cell.Mass;
		cmX += cell.Mass * cell.CmX;
		cmY += cell.Mass * cell.CmY;
	}

	cmX = cmX / mass;
	cmY = cmY / mass;

	cm[0] = (float2)(cmX, cmY);
}

__kernel void CalcForce(global Body* inBodies, int inBodiesLen, global Body* outBodies, global MeshCell* inMesh, int meshTopStart, int meshTopEnd, global int* meshNeighbors, const SimSettings sim, const SPHPreCalc sph, global int* postNeeded)
{
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body and mesh cell from memory.
	Body outBody = inBodies[(a)];
	MeshCell levelCell = inMesh[(outBody.MeshID)];
	MeshCell levelCellParent = inMesh[(levelCell.ParentID)];
	
	// Reset forces.
	float totForce = 0;
	outBody.ForceX = 0.0f;
	outBody.ForceY = 0.0f;
	outBody.Density = 0.0f;
	outBody.Pressure = 0.0f;

	// Resting density.	
	outBody.Density = outBody.Mass * sph.fDensity;

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

				float distX = inBody.PosX - outBody.PosX;
				float distY = inBody.PosY - outBody.PosY;
				float dist = distX * distX + distY * distY;
				float distSqrt = (float)native_sqrt(dist);

				// If this body is within collision/SPH distance.
				if (distSqrt <= sph.kSize)
				{
					// Clamp SPH softening distance.
					dist = max(dist, SPH_SOFTENING);

					// Accumulate density.
					float diff = sph.kSizeSq - dist;
					float fac = sph.fDensity * diff * diff * diff;
					outBody.Density += outBody.Mass * fac;
				}

				// Clamp gravity softening distance.
				dist = max(dist, SOFTENING);
				distSqrt = max(distSqrt, SOFTENING_SQRT);

				// Accumulate body-to-body force.
				float force = inBody.Mass * outBody.Mass / dist;

				totForce += force;
				outBody.ForceX += force * distX / distSqrt;
				outBody.ForceY += force * distY / distSqrt;
			}
		}
	}

	// *** Particle 2 Mesh ***
	// Accumulate force from neighboring cells at each level.
	for (int level = 0; level < sim.MeshLevels; level++)
	{
		// Iterate parent cell neighbors.
		int start = levelCellParent.NeighborStartIdx;
		int len = start + levelCellParent.NeighborCount;

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
					// Calculate the force from the cells center of mass.
					float distX = cell.CmX - outBody.PosX;
					float distY = cell.CmY - outBody.PosY;
					float dist = distX * distX + distY * distY;
					float distSqrt = (float)native_sqrt(dist);
					float force = cell.Mass * outBody.Mass / dist;

					totForce += force;
					outBody.ForceX += force * distX / distSqrt;
					outBody.ForceY += force * distY / distSqrt;
				}
			}
		}

		levelCell = levelCellParent;

		// Move up to next level.
		if (levelCellParent.ParentID != -1)
		{ 
			levelCellParent = inMesh[(levelCellParent.ParentID)];
		}
	}

	// *** Particle 2 Mesh ***
	// Accumulate force from remaining distant cells at the top-most level.
	for (int top = meshTopStart; top < meshTopEnd; top++)
	{
		MeshCell cell = inMesh[(top)];

		if (!IsNeighbor(levelCell, cell))
		{
			float distX = cell.CmX - outBody.PosX;
			float distY = cell.CmY - outBody.PosY;
			float dist = distX * distX + distY * distY;
			float distSqrt = (float)native_sqrt(dist);
			float force = cell.Mass * outBody.Mass / dist;

			totForce += force;
			outBody.ForceX += force * distX / distSqrt;
			outBody.ForceY += force * distY / distSqrt;
		}
	}

	// Calculate pressure from density.
	outBody.Pressure = sim.GasK * outBody.Density;

	if (totForce > outBody.Mass * 4.0f)
	{
		if (!HasFlagB(outBody, INROCHE))
		{
			outBody = SetFlagB(outBody, INROCHE, true);
			postNeeded[0] = 1;
		}
	}

	// Write back to memory.
	outBodies[(a)] = outBody;
}

// Is the specified cell a neighbor of the test cell?
bool IsNeighbor(MeshCell cell, MeshCell testCell)
{
	if (testCell.IdxX > cell.IdxX + -2 && testCell.IdxX < cell.IdxX + 2 && testCell.IdxY > cell.IdxY + -2 && testCell.IdxY < cell.IdxY + 2)
	{
		return true;
	}
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
					float dist = distX * distX + distY * distY;
					float distSqrt = (float)native_sqrt(dist);

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

__kernel void SPHCollisions(global Body* inBodies, int inBodiesLen, global Body* outBodies, global MeshCell* inMesh, global int* meshNeighbors, global float2* centerMass, const SimSettings sim, const SPHPreCalc sph, global int* postNeeded)
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
					float dist = distX * distX + distY * distY;

					// Calc the distance and check for collision.
					float colDist = (sph.kSize * 0.5f) * 2.0f;
					if (dist <= colDist * colDist)
					{
						// Handle exact overlaps.
						if (dist == 0)
						{
							outBody.PosX += (outBody.UID + 1) * SPH_SOFTENING;
							outBody.PosY += (outBody.UID + 1) * SPH_SOFTENING;
						}

						dist = max(dist, SPH_SOFTENING);

						// Only do SPH collision if both bodies are in roche.
						// SPH collision.
						if (HasFlagB(outBody, INROCHE) && HasFlagB(inBody, INROCHE))
						{
							float distSqrt = (float)native_sqrt(dist);
							distSqrt = max(distSqrt, SPH_SOFTENING);

							float kDiff = sph.kSize - distSqrt;
						
							// Pressure force
							float pressScalar = outBody.Mass * (outBody.Pressure + inBody.Pressure) / (2.0f * outBody.Density);
							float pressGrad = sph.fPressure * kDiff * kDiff / distSqrt;

							outBody.ForceX += (distX * pressGrad) * pressScalar;
							outBody.ForceY += (distY * pressGrad) * pressScalar;

							// Viscosity force
							float viscLaplace = sph.fViscosity * kDiff;
							float viscScalar = inBody.Mass * viscLaplace * sim.Viscosity / inBody.Density;

							outBody.ForceX += (inBody.VeloX - outBody.VeloX) * viscScalar;
							outBody.ForceY += (inBody.VeloY - outBody.VeloY) * viscScalar;

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
	float dist = fast_distance(cm, (float2)(outBody.PosX, outBody.PosY));

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

	outBody.VeloX += colMass * forceX;
	outBody.VeloY += colMass * forceY;

	// Don't increase size of black holes.
	if (!HasFlagB(outBody, BLACKHOLE))
	{
		float a1 = pow((outBody.Size * 0.5f), 2.0f);
		float a2 = pow((bodyB.Size * 0.5f), 2.0f);
		float area = a1 + a2;
		outBody.Size = (float)native_sqrt(area) * 2.0f;
	}

	outBody.Mass += bodyB.Mass;

	return outBody;
}