
struct Body
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
};

struct MeshCell
{
	int ID;
	float LocX;
	float LocY;
	int IdxX;
	int IdxY;
	int Mort;
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
};

struct GridInfo
{
	int OffsetX;
	int OffsetY;
	int MinX;
	int MinY;
	int MaxX;
	int MaxY;
	int Columns;
	int Rows;
	int Size;
	int IndexOffset;
};


int IsNear(struct MeshCell testCell, struct MeshCell neighborCell);
struct Body CollideBodies(struct Body master, struct Body slave, float colMass, float forceX, float forceY);

__kernel void FixOverlaps(global struct Body* inBodies, int inBodiesLen, global struct Body* outBodies)
{
	int i = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (i >= inBodiesLen)
	{
		return;
	}

	struct Body bodyA = inBodies[i];

	for (int j = 0; j < inBodiesLen; j++)
	{
		if (i != j)
		{
			struct Body bodyB = inBodies[j];
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

__kernel void ClearGrid(global int* gridIdx, global struct MeshCell* mesh, int meshLen)
{
	int m = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (m >= meshLen)
	{
		return;
	}

	gridIdx[mesh[m].GridIdx] = 0;
}

__kernel void PopGrid(global int* gridIdx, global struct GridInfo* gridInfo, global struct MeshCell* mesh, int meshLen)
{
	int m = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (m >= meshLen)
	{
		return;
	}

	struct MeshCell cell = mesh[m];

	int level = cell.Level;

	struct GridInfo grid = gridInfo[level];

	int column = cell.IdxX + grid.OffsetX;
	int row = cell.IdxY + grid.OffsetY;
	int idx = (row * grid.Columns) + column + row;

	idx += grid.IndexOffset;
	cell.GridIdx = idx;

	if (cell.ID == 0)
	{
		gridIdx[idx] = -1;
	}
	else
	{
		gridIdx[idx] = cell.ID;
	}

	mesh[m] = cell;
}

__kernel void BuildNeighbors(global struct MeshCell* mesh, int meshLen, global struct GridInfo* gridInfo, global int* gridIdx, int gridIdxLen, global int* neighborIndex)
{
	int m = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (m >= meshLen)
	{
		return;
	}

	int offset = m * 9;
	int count = 0;
	struct MeshCell cell = mesh[m];
	struct GridInfo gInfo = gridInfo[cell.Level];
	int columns = gInfo.Columns;

	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			int offIdx = cell.GridIdx - gInfo.IndexOffset;
			int localIdx = offIdx + ((x * columns) + (y + x));

			if (localIdx > 0 && localIdx < gInfo.Size)
			{
				int globalIdx = localIdx + gInfo.IndexOffset;

				if (gridIdx[globalIdx] > 0)
				{
					neighborIndex[(offset + count)] = gridIdx[globalIdx];
					count++;
				}
				else if (gridIdx[globalIdx] == -1)
				{
					neighborIndex[(offset + count)] = 0;
					count++;
				}
			}
		}
	}

	for (int i = (offset + count); i < offset + 9; i++)
	{
		neighborIndex[i] = -1;
	}

	cell.NeighborStartIdx = offset;
	cell.NeighborCount = count;

	mesh[m] = cell;
}


__kernel void CalcForce(global struct Body* inBodies, int inBodiesLen, global struct Body* outBodies, global struct MeshCell* inMesh, int inMeshLen, global int* meshNeighbors, float dt, int topLevel, global int* levelIdx, int levelIdxLen0)
{
	float GAS_K = 0.3f;
	float FLOAT_EPSILON = 1.192093E-07f;

	int a = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body and mesh cell from memory.
	struct Body outBody = inBodies[(a)];
	struct MeshCell bodyCell = inMesh[(outBody.MeshID)];
	struct MeshCell levelCell = bodyCell;
	struct MeshCell levelCellParent = inMesh[(bodyCell.ParentID)];

	// Reset forces.
	outBody.ForceTot = 0.0f;
	outBody.ForceX = 0.0f;
	outBody.ForceY = 0.0f;
	outBody.Density = 0.0f;
	outBody.Pressure = 0.0f;

	float ksize = 1.0f;
	float ksizeSq = 1.0f;
	float factor = 1.566682f;
	float softening = 0.04f;

	outBody.Density = outBody.Mass * factor;

	for (int level = 0; level < topLevel; level++)
	{
		// Iterate parent cell neighbors.
		int start = levelCellParent.NeighborStartIdx;
		int len = start + levelCellParent.NeighborCount;

		for (int nc = start; nc < len; nc++)
		{
			int nId = meshNeighbors[(nc)];
			struct MeshCell nCell = inMesh[(nId)];

			// Iterate neighbor child cells.
			int childStartIdx = nCell.ChildStartIdx;
			int childLen = childStartIdx + nCell.ChildCount;
			for (int c = childStartIdx; c < childLen; c++)
			{
				// Make sure the current cell index is not a neighbor or this body's cell.
				if (c != outBody.MeshID)
				{
					struct MeshCell cell = inMesh[(c)];

					if (IsNear(levelCell, cell) == 0)
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
	for (int top = levelIdx[(topLevel)]; top < inMeshLen; top++)
	{
		struct MeshCell cell = inMesh[(top)];

		if (IsNear(levelCell, cell) == 0)
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
		struct MeshCell cell = inMesh[(nId)];

		// Iterate the bodies within the cell.
		// Read from body array at the correct location.
		int mbStart = cell.BodyStartIdx;
		int mbLen = cell.BodyCount + mbStart;
		for (int mb = mbStart; mb < mbLen; mb++)
		{
			// Save us from ourselves.
			if (mb != a)
			{
				struct Body inBody = inBodies[(mb)];

				float distX = inBody.PosX - outBody.PosX;
				float distY = inBody.PosY - outBody.PosY;
				float dist = distX * distX + distY * distY;

				// If this body is within collision/SPH distance.
				if (dist <= ksize)
				{
					// Clamp SPH softening distance.
					if (dist < FLOAT_EPSILON)
					{
						dist = FLOAT_EPSILON;
					}

					// Accumulate density.
					float diff = ksizeSq - dist;
					float fac = factor * diff * diff * diff;
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

	// Write back to memory.
	outBodies[(a)] = outBody;
}

// Is the specified cell a neighbor of the test cell?
int IsNear(struct MeshCell cell, struct MeshCell testCell)
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

__kernel void CalcCollisions(global struct Body* inBodies, int inBodiesLen, global struct Body* outBodies, global struct MeshCell* inMesh, global int* meshNeighbors, float dt, float viscosity, float2 centerMass, float cullDistance, int collisions)
{
	// Get index for the current body.
	int a = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy current body from memory.
	struct Body outBody = inBodies[(a)];

	// Copy this body's mesh cell from memory.
	struct MeshCell bodyCell = inMesh[(outBody.MeshID)];

	if (collisions == 1)
	{
		// Iterate neighbor cells.
		for (int i = bodyCell.NeighborStartIdx; i < bodyCell.NeighborStartIdx + bodyCell.NeighborCount; i++)
		{
			// Get the neighbor cell from the index.
			int nId = meshNeighbors[(i)];
			struct MeshCell cell = inMesh[(nId)];

			// Iterate the neighbor cell bodies.
			int mbStart = cell.BodyStartIdx;
			int mbLen = cell.BodyCount + mbStart;
			for (int mb = mbStart; mb < mbLen; mb++)
			{
				// Double tests are bad.
				if (mb != a)
				{
					struct Body inBody = inBodies[(mb)];

					float distX = outBody.PosX - inBody.PosX;
					float distY = outBody.PosY - inBody.PosY;
					float dist = distX * distX + distY * distY;

					// Calc the distance and check for collision.
					float colDist = outBody.Size * 0.5f + inBody.Size * 0.5f;
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
							float kernelSize = 1.0f;

							if (dist < FLOAT_EPSILON)
							{
								distSqrt = FLOAT_EPSILONSQRT;
							}

							// Pressure force
							float scalar = outBody.Mass * (outBody.Pressure + inBody.Pressure) / (2.0f * outBody.Density);
							float gradFactor = -10442.157f * (kernelSize - distSqrt) * (kernelSize - distSqrt) / distSqrt;

							float gradX = distX * gradFactor;
							float gradY = distY * gradFactor;

							gradX *= scalar;
							gradY *= scalar;

							outBody.ForceX -= gradX;
							outBody.ForceY -= gradY;

							// Viscosity force
							float visc_laplace = 14.323944f * (kernelSize - distSqrt);
							float visc_scalar = inBody.Mass * visc_laplace * viscosity * 1.0f / inBody.Density;

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
	outBody.VeloX += dt * outBody.ForceX / outBody.Mass;
	outBody.VeloY += dt * outBody.ForceY / outBody.Mass;
	outBody.PosX += dt * outBody.VeloX;
	outBody.PosY += dt * outBody.VeloY;

	if (outBody.Lifetime > 0.0f)
	{
		outBody.Age += dt * 4.0f;
	}

	// Cull distant bodies.
	float distX = centerMass.x - outBody.PosX;
	float distY = centerMass.y - outBody.PosY;
	float dist = distX * distX + distY * distY;

	if (dist > cullDistance * cullDistance)
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

struct Body CollideBodies(struct Body bodyA, struct Body bodyB, float colMass, float forceX, float forceY)
{
	struct Body outBody = bodyA;

	outBody.VeloX += colMass * forceX;
	outBody.VeloY += colMass * forceY;

	if (outBody.Flag != 1)
	{
		float a1 = 3.141593f * (float)pow((outBody.Size * 0.5f), 2.0f);
		float a2 = 3.141593f * (float)pow((bodyB.Size * 0.5f), 2.0f);
		float area = a1 + a2;
		outBody.Size = (float)native_sqrt((((area / 3.141593f)) * 2.0f));
	}

	outBody.Mass += bodyB.Mass;

	return outBody;
}