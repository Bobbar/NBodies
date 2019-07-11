
typedef struct __attribute__((packed)) Body
{
	float PosX;
	float PosY;
	float Mass;
	float VeloX;
	float VeloY;
	float ForceX;
	float ForceY;
	float ForceTot;
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
	int CollisionsOn;
	int MeshLevels;
	int CellSizeExponent;

} SimSettings;

constant int BLACKHOLE = 1;
constant int ISEXPLOSION = 2;
constant int CULLED = 4;
constant int INROCHE = 8;

int IsNeighbor(MeshCell testCell, MeshCell neighborCell);
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


__kernel void FixOverlaps(global  Body* inBodies, int inBodiesLen, global  Body* outBodies)
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

__kernel void ClearGrid(global int* gridIdx, int passStride, int passOffset, global  MeshCell* mesh, int meshLen)
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

__kernel void PopGrid(global int* gridIdx, int passStride, int passOffset, global GridInfo* gridInfo, global  MeshCell* mesh, int meshLen)
{
	int m = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (m >= meshLen)
	{
		return;
	}

	MeshCell cell = mesh[m];

	int level = cell.Level;

	GridInfo grid = gridInfo[level];

	int column = cell.IdxX + grid.OffsetX;
	int row = cell.IdxY + grid.OffsetY;
	int bucket = (row * grid.Columns) + column + row;

	bucket += grid.IndexOffset;

	cell.GridIdx = bucket;

	bucket -= passOffset;

	if (bucket >= 0 && bucket < passStride)
	{
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

__kernel void BuildNeighbors(global  MeshCell* mesh, int meshLen, global GridInfo* gridInfo, global int* gridIdx, int passStride, int passOffset, global int* neighborIndex)
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

		cell.NeighborStartIdx = offset;
		cell.NeighborCount = count;
	}

	mesh[m] = cell;
}

__kernel void BuildBottom(global  Body* inBodies, global  Body* outBodies, global  MeshCell* mesh, int meshLen, global int* cellIdx, int cellSizeExp)
{
	int m = get_global_id(0);

	if (m >= meshLen)
		return;

	int firstIdx = cellIdx[m];
	Body firstBody = inBodies[firstIdx];

	MeshCell newCell;
	newCell.IdxX = (int)firstBody.PosX >> cellSizeExp;
	newCell.IdxY = (int)firstBody.PosY >> cellSizeExp;
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

__kernel void BuildTop(global  MeshCell* mesh, int len, global int* cellIdx, int cellSizeExp, int levelOffset, int meshOffset, int readOffset, int level)
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

__kernel void CalcCenterOfMass(global  MeshCell* inMesh, global float2* cm, int start, int end)
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

__kernel void CalcForce(global  Body* inBodies, int inBodiesLen, global  Body* outBodies, global  MeshCell* inMesh, int inMeshLen, global int* meshNeighbors, global int* levelIdx, const SimSettings sim, const SPHPreCalc sph, global int* postNeeded)
{
	float GAS_K = 0.3f;
	float FLOAT_EPSILON = 1.192093E-07f;
	float softening = 0.04f;

	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body and mesh cell from memory.
	Body outBody = inBodies[(a)];
	MeshCell levelCell = inMesh[(outBody.MeshID)];
	MeshCell levelCellParent = inMesh[(levelCell.ParentID)];

	// Reset forces.
	outBody.ForceTot = 0.0f;
	outBody.ForceX = 0.0f;
	outBody.ForceY = 0.0f;
	outBody.Density = 0.0f;
	outBody.Pressure = 0.0f;

	// Resting density.	
	outBody.Density = outBody.Mass * sph.fDensity;

	// Accumulate forces from all bodies within neighboring cells. [THIS INCLUDES THE BODY'S OWN CELL]
	// Read from the flattened mesh-neighbor index at the correct location.
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
					dist = max(dist, FLOAT_EPSILON);
					

					// Accumulate density.
					float diff = sph.kSizeSq - dist;
					float fac = sph.fDensity * diff * diff * diff;
					outBody.Density += outBody.Mass * fac;
				}

				// Clamp gravity softening distance.
				dist = max(dist, softening);
				
				// Accumulate body-to-body force.
				float force = inBody.Mass * outBody.Mass / dist;

				outBody.ForceTot += force;
				outBody.ForceX += force * distX / distSqrt;
				outBody.ForceY += force * distY / distSqrt;
			}
		}
	}


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
				// Make sure the current cell index is not a neighbor or this body's cell.
				if (c != outBody.MeshID)
				{
					MeshCell cell = inMesh[(c)];

					if (IsNeighbor(levelCell, cell) == 0)
					{
						// Calculate the force from the cells center of mass.
						float distX = cell.CmX - outBody.PosX;
						float distY = cell.CmY - outBody.PosY;
						float dist = distX * distX + distY * distY;
						float distSqrt = (float)native_sqrt(dist);
						float force = (float)cell.Mass * outBody.Mass / dist;

						outBody.ForceTot += force;
						outBody.ForceX += force * distX / distSqrt;
						outBody.ForceY += force * distY / distSqrt;
					}
				}
			}
		}

		// Move up to next level.
		levelCell = levelCellParent;
		levelCellParent = inMesh[(levelCellParent.ParentID)];
	}

	// Iterate the top level cells.
	for (int top = levelIdx[(sim.MeshLevels)]; top < inMeshLen; top++)
	{
		MeshCell cell = inMesh[(top)];

		if (IsNeighbor(levelCell, cell) == 0)
		{
			float distX = cell.CmX - outBody.PosX;
			float distY = cell.CmY - outBody.PosY;
			float dist = distX * distX + distY * distY;
			float distSqrt = (float)native_sqrt(dist);
			float force = (float)cell.Mass * outBody.Mass / dist;

			outBody.ForceTot += force;
			outBody.ForceX += force * distX / distSqrt;
			outBody.ForceY += force * distY / distSqrt;
		}
	}



	// Calculate pressure from density.
	outBody.Pressure = GAS_K * outBody.Density;

	if (outBody.ForceTot > outBody.Mass * 4.0f)
	{
		outBody = SetFlagB(outBody, INROCHE, true);
		postNeeded[0] = 1;
	}


	// Write back to memory.
	outBodies[(a)] = outBody;
}

// Is the specified cell a neighbor of the test cell?
int IsNeighbor(MeshCell cell, MeshCell testCell)
{
	int result = 0;

	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			if (testCell.IdxX == cell.IdxX + x && testCell.IdxY == cell.IdxY + y)
			{
				result = 1;
			}
		}
	}

	return result;
}

// Collision pass for bodies larger than thier parent cells.
__kernel void CalcCollisionsLarge(global  Body* inBodies, int inBodiesLen, global  Body* outBodies, global  MeshCell* inMesh, global int* meshNeighbors, int collisions, global int* postNeeded)
{
	// Get index for the current body.
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body from memory.
	Body outBody = inBodies[(a)];

	if (collisions == 1)
	{
		// Only proceed for bodies larger than 1 unit.
		if (outBody.Size <= 1.0f)
		{
			outBodies[a] = outBody;
			return;
		}

		// Get the current parent cell.
		MeshCell parentCell = inMesh[(outBody.MeshID)];
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
			int nId = meshNeighbors[(i)];
			MeshCell nCell = inMesh[(nId)];

			// Iterate all the bodies within each neighboring cell.
			int mbStart = nCell.BodyStartIdx;
			int mbLen = nCell.BodyCount + mbStart;
			for (int mb = mbStart; mb < mbLen; mb++)
			{
				// Save us from ourselves.
				if (mb != a)
				{
					Body inBody = inBodies[(mb)];

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
							outBodies[(mb)].Flag = SetFlag(outBodies[(mb)].Flag, CULLED, true);
							postNeeded[0] = 1;
						}
						else if (outBody.Mass < inBody.Mass) // We're smaller, so we must go away.
						{
							outBody.Flag = SetFlag(outBody.Flag, CULLED, true);
							outBodies[(mb)] = CollideBodies(inBody, outBody, colMass, forceX, forceY);
							postNeeded[0] = 1;
						}
						else if (outBody.Mass == inBody.Mass) // If we are the same size, use a different metric.
						{
							// Our UID is more gooder, eat the other guy.
							if (outBody.UID > inBody.UID)
							{
								outBody = CollideBodies(outBody, inBody, colMass, forceX, forceY);
								outBodies[(mb)].Flag = SetFlag(outBodies[(mb)].Flag, CULLED, true);
								postNeeded[0] = 1;
							}
							else // Our UID is inferior, we must go away.
							{
								outBody.Flag = SetFlag(outBody.Flag, CULLED, true);
								outBodies[(mb)] = CollideBodies(inBody, outBody, colMass, forceX, forceY);
								postNeeded[0] = 1;
							}
						}
					}
				}
			}
		}
	}
	outBodies[(a)] = outBody;
}


__kernel void CalcCollisions(global  Body* inBodies, int inBodiesLen, global  Body* outBodies, global  MeshCell* inMesh, global int* meshNeighbors, global float2* centerMass, const SimSettings sim, const SPHPreCalc sph, global int* postNeeded)
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
						// We know we have a collision, so go ahead and do the expensive square root now.
						float distSqrt = (float)native_sqrt(dist);

						// If both bodies are in Roche, we do SPH physics.
						// Otherwise, an elastic collision and merge is done.

						// SPH collision.
						if (HasFlagB(outBody, INROCHE) && HasFlagB(inBody, INROCHE))
						{
							float FLOAT_EPSILON = 1.192092896e-07f;
							float FLOAT_EPSILONSQRT = 3.45267e-11f;

							distSqrt = max(distSqrt, FLOAT_EPSILONSQRT);
							
							float kDiff = sph.kSize - distSqrt;

							// Pressure force
							float scalar = outBody.Mass * (outBody.Pressure + inBody.Pressure) / (2.0f * outBody.Density);
							float gradFactor = -sph.fPressure * kDiff * kDiff / distSqrt;

							float gradX = distX * gradFactor;
							float gradY = distY * gradFactor;

							gradX *= scalar;
							gradY *= scalar;

							outBody.ForceX -= gradX;
							outBody.ForceY -= gradY;

							// Viscosity force
							float visc_laplace = sph.fViscosity * kDiff;
							float visc_scalar = inBody.Mass * visc_laplace * sim.Viscosity * 1.0f / inBody.Density;

							float viscVelo_diffX = inBody.VeloX - outBody.VeloX;
							float viscVelo_diffY = inBody.VeloY - outBody.VeloY;

							viscVelo_diffX *= visc_scalar;
							viscVelo_diffY *= visc_scalar;

							outBody.ForceX += viscVelo_diffX;
							outBody.ForceY += viscVelo_diffY;

						}
						// Elastic collision.
						else if (HasFlagB(outBody, INROCHE) && !HasFlagB(inBody, INROCHE))
						{
							outBody.Flag = SetFlag(outBody.Flag, CULLED, true);
						}
						else
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
								postNeeded[0] = 1;
							}
							else if (outBody.Mass < inBody.Mass) // We're smaller, so we must go away.
							{
								outBody.Flag = SetFlag(outBody.Flag, CULLED, true);
								postNeeded[0] = 1;
							}
							else if (outBody.Mass == inBody.Mass) // If we are the same size, use a different metric.
							{
								// Our UID is more gooder, eat the other guy.
								if (outBody.UID > inBody.UID)
								{
									outBody = CollideBodies(outBody, inBody, colMass, forceX, forceY);
									postNeeded[0] = 1;
								}
								else // Our UID is inferior, we must go away.
								{
									outBody.Flag = SetFlag(outBody.Flag, CULLED, true);
									postNeeded[0] = 1;
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
	outBody.PosX += sim.DeltaTime * outBody.VeloX;
	outBody.PosY += sim.DeltaTime * outBody.VeloY;

	if (outBody.Lifetime > 0.0f)
	{
		outBody.Lifetime -= sim.DeltaTime * 4.0f;
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