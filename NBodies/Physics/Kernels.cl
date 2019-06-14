
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
	int Visible;
	int InRoche;
	int Flag;
	int UID;
	float Density;
	float Pressure;
	int IsExplosion;
	float Lifetime;
	float Age;
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
	float DeltaTime;
	float Viscosity;
	//float2 CenterMass;
	float CullDistance;
	bool CollisionsOn;
	int MeshLevels;
	int CellSizeExponent;

} SimSettings;

int IsNeighbor(MeshCell testCell, MeshCell neighborCell);
Body CollideBodies(Body master, Body slave, float colMass, float forceX, float forceY);


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

__kernel void BuildBottom(global  Body* inBodies, global  Body* outBodies, global  MeshCell* mesh, int meshLen, global int* cellIdx, global float2* locIdx, int cellSize)
{
	int m = get_global_id(0);

	if (m >= meshLen)
		return;

	MeshCell newCell;
	float2 idx = locIdx[m];

	newCell.IdxX = (int)idx.x;
	newCell.IdxY = (int)idx.y;
	newCell.Size = cellSize;
	newCell.BodyCount = 0;
	newCell.ChildCount = 0;
	newCell.ID = 0;
	newCell.Level = 0;
	newCell.ParentID = -1;
	newCell.CmX = 0;
	newCell.CmY = 0;
	newCell.Mass = 0;

	for (int i = cellIdx[m]; i < cellIdx[m + 1]; i++)
	{
		Body body = inBodies[i];
		newCell.Mass += body.Mass;
		newCell.CmX += body.Mass * body.PosX;
		newCell.CmY += body.Mass * body.PosY;
		newCell.BodyCount++;

		body.MeshID = m;

		outBodies[i] = body;
	}

	newCell.ID = m;
	newCell.BodyStartIdx = cellIdx[m];

	newCell.CmX = newCell.CmX / (float)newCell.Mass;
	newCell.CmY = newCell.CmY / (float)newCell.Mass;

	mesh[m] = newCell;

}

__kernel void BuildTop(global  MeshCell* mesh, int len, global int* cellIdx, global float2* locIdx, int cellSize, int levelOffset, int meshOffset, int readOffset, int level)
{
	int m = get_global_id(0);

	if (m >= len)
		return;

	int locIdxOff = m + readOffset;
	int cellIdxOff = locIdxOff + (level - 1);
	int newIdx = m + meshOffset;

	MeshCell newCell;

	float2 idx = locIdx[locIdxOff];
	newCell.IdxX = (int)idx.x;
	newCell.IdxY = (int)idx.y;
	newCell.Size = cellSize;
	newCell.ChildStartIdx = cellIdx[cellIdxOff] + levelOffset;
	newCell.ChildCount = 0;
	newCell.ID = newIdx;
	newCell.Level = level;
	newCell.BodyStartIdx = 0;
	newCell.BodyCount = 0;
	newCell.ParentID = -1;
	newCell.CmX = 0;
	newCell.CmY = 0;
	newCell.Mass = 0;


	for (int i = cellIdx[cellIdxOff]; i < cellIdx[cellIdxOff + 1]; i++)
	{
		MeshCell child = mesh[i + levelOffset];
		newCell.Mass += child.Mass;
		newCell.CmX += (float)child.Mass * child.CmX;
		newCell.CmY += (float)child.Mass * child.CmY;
		newCell.ChildCount++;
		newCell.BodyCount += child.BodyCount;

		if (newCell.ChildCount == 1)
			newCell.BodyStartIdx = child.BodyStartIdx;

		child.ParentID = newIdx;
		mesh[i + levelOffset] = child;
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

__kernel void CalcForce(global  Body* inBodies, int inBodiesLen, global  Body* outBodies, global  MeshCell* inMesh, int inMeshLen, global int* meshNeighbors, global int* levelIdx, const SimSettings sim, const SPHPreCalc sph)
{
	float GAS_K = 0.3f;
	float FLOAT_EPSILON = 1.192093E-07f;
	float softening = 0.04f;

	int a = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body and mesh cell from memory.
	Body outBody = inBodies[(a)];
	MeshCell bodyCell = inMesh[(outBody.MeshID)];
	MeshCell levelCell = bodyCell;
	MeshCell levelCellParent = inMesh[(bodyCell.ParentID)];

	// Reset forces.
	outBody.ForceTot = 0.0f;
	outBody.ForceX = 0.0f;
	outBody.ForceY = 0.0f;
	outBody.Density = 0.0f;
	outBody.Pressure = 0.0f;

	// Resting density.	
	outBody.Density = outBody.Mass * sph.fDensity;

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

	// Accumulate forces from all bodies within neighboring cells. [THIS INCLUDES THE BODY'S OWN CELL]
	// Read from the flattened mesh-neighbor index at the correct location.
	for (int n = bodyCell.NeighborStartIdx; n < bodyCell.NeighborStartIdx + bodyCell.NeighborCount; n++)
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

				// If this body is within collision/SPH distance.
				if (dist <= sph.kSize)
				{
					// Clamp SPH softening distance.
					if (dist < FLOAT_EPSILON)
					{
						dist = FLOAT_EPSILON;
					}

					// Accumulate density.
					float diff = sph.kSizeSq - dist;
					float fac = sph.fDensity * diff * diff * diff;
					outBody.Density += outBody.Mass * fac;
				}

				// Clamp gravity softening distance.
				if (dist < softening)
				{
					dist = softening;
				}

				// Accumulate body-to-body force.
				float distSqrt = (float)native_sqrt(dist);
				float force = inBody.Mass * outBody.Mass / dist;

				outBody.ForceTot += force;
				outBody.ForceX += force * distX / distSqrt;
				outBody.ForceY += force * distY / distSqrt;
			}
		}
	}

	// Calculate pressure from density.
	outBody.Pressure = GAS_K * outBody.Density;

	if (outBody.ForceTot > outBody.Mass * 4.0f & outBody.Flag == 0)
	{
		outBody.InRoche = 1;
	}
	else if (outBody.ForceTot * 2.0f < outBody.Mass * 4.0f)
	{
		outBody.InRoche = 0;
	}
	else if (outBody.IsExplosion == 1)
	{
		outBody.InRoche = 1;
	}

	if (outBody.Flag == 2)
	{
		outBody.InRoche = 1;
	}

	if (outBody.Size <= 1.1f)
		outBody.InRoche = 1;

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
__kernel void CalcCollisionsLarge(global  Body* inBodies, int inBodiesLen, global  Body* outBodies, global  MeshCell* inMesh, global int* meshNeighbors, int collisions)
{
	// Get index for the current body.
	int a = get_local_size(0) * get_group_id(0) + get_local_id(0);

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
							outBodies[(mb)].Visible = 0;
						}
						else if (outBody.Mass < inBody.Mass) // We're smaller, so we must go away.
						{
							outBody.Visible = 0;
							outBodies[(mb)] = CollideBodies(inBody, outBody, colMass, forceX, forceY);
						}
						else if (outBody.Mass == inBody.Mass) // If we are the same size, use a different metric.
						{
							// Our UID is more gooder, eat the other guy.
							if (outBody.UID > inBody.UID)
							{
								outBody = CollideBodies(outBody, inBody, colMass, forceX, forceY);
								outBodies[(mb)].Visible = 0;
							}
							else // Our UID is inferior, we must go away.
							{
								outBody.Visible = 0;
								outBodies[(mb)] = CollideBodies(inBody, outBody, colMass, forceX, forceY);
							}
						}
					}
				}
			}
		}
	}
	outBodies[(a)] = outBody;
}


__kernel void CalcCollisions(global  Body* inBodies, int inBodiesLen, global  Body* outBodies, global  MeshCell* inMesh, global int* meshNeighbors, global float2* centerMass, const SimSettings sim, const SPHPreCalc sph)
{
	// Get index for the current body.
	int a = get_local_size(0) * get_group_id(0) + get_local_id(0);

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
					float colDist = sph.kSize * 0.5f + sph.kSize * 0.5f;
					if (dist <= colDist * colDist)
					{
						// We know we have a collision, so go ahead and do the expensive square root now.
						float distSqrt = (float)native_sqrt(dist);

						// If both bodies are in Roche, we do SPH physics.
						// Otherwise, an elastic collision and merge is done.

						// SPH collision.
						if (outBody.InRoche == 1 && inBody.InRoche == 1)
						{
							float FLOAT_EPSILON = 1.192092896e-07f;
							float FLOAT_EPSILONSQRT = 3.45267E-11f;

							if (dist < FLOAT_EPSILON)
							{
								distSqrt = FLOAT_EPSILONSQRT;
							}

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
						else if (outBody.InRoche == 1 && inBody.InRoche == 0)
						{
							outBody.Visible = 0;
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
							}
							else if (outBody.Mass < inBody.Mass) // We're smaller, so we must go away.
							{
								outBody.Visible = 0;
							}
							else if (outBody.Mass == inBody.Mass) // If we are the same size, use a different metric.
							{
								// Our UID is more gooder, eat the other guy.
								if (outBody.UID > inBody.UID)
								{
									outBody = CollideBodies(outBody, inBody, colMass, forceX, forceY);
								}
								else // Our UID is inferior, we must go away.
								{
									outBody.Visible = 0;
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
		outBody.Age += sim.DeltaTime * 4.0f;
	}

	// Cull distant bodies.
	float2 cm = centerMass[0];
	float distX = cm.x - outBody.PosX;
	float distY = cm.y - outBody.PosY;
	float dist = distX * distX + distY * distY;
	float dsqrt = native_sqrt(dist);

	if (dsqrt > sim.CullDistance)
		outBody.Visible = 0;

	// Cull expired bodies.
	if (outBody.Age > outBody.Lifetime)
	{
		outBody.Visible = 0;
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

	if (outBody.Flag != 1)
	{
		float a1 = pow((outBody.Size * 0.5f), 2.0f);
		float a2 = pow((bodyB.Size * 0.5f), 2.0f);
		float area = a1 + a2;
		outBody.Size = (float)native_sqrt(area) * 2.0f;
	}

	outBody.Mass += bodyB.Mass;

	return outBody;
}