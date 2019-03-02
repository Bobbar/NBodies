#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif
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
	double Mass;
	int Size;
	int BodyStartIdx;
	int BodyCount;
	int NeighborStartIdx;
	int NeighborCount;
	int ChildStartIdx;
	int ChildCount;
	int ParentID;
	int Level;
};


// NBodies.Physics.CUDAFloat
__kernel  void CalcForce(global struct Body* inBodies, int inBodiesLen0, global struct Body* outBodies, int outBodiesLen0, global struct MeshCell* inMesh, int inMeshLen0, global int* meshNeighbors, int meshNeighborsLen0, float dt, int topLevel, global int* levelIdx, int levelIdxLen0);
// NBodies.Physics.CUDAFloat
int IsNear(struct MeshCell testCell, struct MeshCell neighborCell);
// NBodies.Physics.CUDAFloat
__kernel  void CalcCollisions(global struct Body* inBodies, int inBodiesLen0, global struct Body* outBodies, int outBodiesLen0, global struct MeshCell* inMesh, int inMeshLen0, global int* meshNeighbors, int meshNeighborsLen0, float dt, float viscosity);
// NBodies.Physics.CUDAFloat
struct Body CollideBodies(struct Body master, struct Body slave, float colMass, float forceX, float forceY);

// NBodies.Physics.CUDAFloat
__kernel  void CalcForce(global struct Body* inBodies, int inBodiesLen0, global struct Body* outBodies, int outBodiesLen0, global struct MeshCell* inMesh, int inMeshLen0, global int* meshNeighbors, int meshNeighborsLen0, float dt, int topLevel, global int* levelIdx, int levelIdxLen0)
{
	float GAS_K = 0.3f;
	float FLOAT_EPSILON = 1.192093E-07f;

	int a = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (a > inBodiesLen0 - 1)
	{
		return;
	}

	// Copy current body and mesh cell from memory.
	struct Body body = inBodies[(a)];
	struct MeshCell bodyCell = inMesh[(body.MeshID)];
	struct MeshCell levelCell = bodyCell;
	struct MeshCell levelCellParent = inMesh[(bodyCell.ParentID)];

	// Reset forces.
	body.ForceTot = 0.0f;
	body.ForceX = 0.0f;
	body.ForceY = 0.0f;
	body.Density = 0.0f;
	body.Pressure = 0.0f;


	float ksize = 1.0f;
	float ksizeSq = 1.0f;
	float factor = 1.566682f;

	body.Density = body.Mass * factor;
	for (int i = 0; i < topLevel; i++)
	{
		int expr_D9 = levelCellParent.NeighborStartIdx;
		int num7 = expr_D9 + levelCellParent.NeighborCount;
		for (int j = expr_D9; j < num7; j++)
		{
			int num8 = meshNeighbors[(j)];
			struct MeshCell meshCell3 = inMesh[(num8)];
			int expr_103 = meshCell3.ChildStartIdx;
			int num9 = expr_103 + meshCell3.ChildCount;
			for (int k = expr_103; k < num9; k++)
			{
				if (k != body.MeshID)
				{
					struct MeshCell meshCell4 = inMesh[(k)];
					if (IsNear(levelCell, meshCell4) == 0)
					{
						float num10 = meshCell4.CmX - body.PosX;
						float num11 = meshCell4.CmY - body.PosY;
						float num12 = num10 * num10 + num11 * num11;
						float num13 = (float)half_sqrt((float)num12);
						float num14 = (float)meshCell4.Mass * body.Mass / num12;
						body.ForceTot += num14;
						body.ForceX += num14 * num10 / num13;
						body.ForceY += num14 * num11 / num13;
					}
				}
			}
		}
		levelCell = levelCellParent;
		levelCellParent = inMesh[(levelCellParent.ParentID)];
	}
	for (int l = levelIdx[(topLevel)]; l < inMeshLen0; l++)
	{
		struct MeshCell meshCell5 = inMesh[(l)];
		if (IsNear(levelCell, meshCell5) == 0)
		{
			float num10 = meshCell5.CmX - body.PosX;
			float num11 = meshCell5.CmY - body.PosY;
			float num12 = num10 * num10 + num11 * num11;
			float num13 = (float)half_sqrt((float)num12);
			float num14 = (float)meshCell5.Mass * body.Mass / num12;
			body.ForceTot += num14;
			body.ForceX += num14 * num10 / num13;
			body.ForceY += num14 * num11 / num13;
		}
	}
	for (int m = bodyCell.NeighborStartIdx; m < bodyCell.NeighborStartIdx + bodyCell.NeighborCount; m++)
	{
		int num15 = meshNeighbors[(m)];
		struct MeshCell expr_2D0 = inMesh[(num15)];
		int bodyStartIdx = expr_2D0.BodyStartIdx;
		int num16 = expr_2D0.BodyCount + bodyStartIdx;
		for (int n = bodyStartIdx; n < num16; n++)
		{
			if (n != a)
			{
				struct Body expr_2FC = inBodies[(n)];
				float num10 = expr_2FC.PosX - body.PosX;
				float num11 = expr_2FC.PosY - body.PosY;
				float num12 = num10 * num10 + num11 * num11;
				if (num12 <= ksize)
				{
					if (num12 < FLOAT_EPSILON)
					{
						num12 = FLOAT_EPSILON;
					}
					float num17 = ksizeSq - num12;
					float num18 = factor * num17 * num17 * num17;
					body.Density += body.Mass * num18;
				}
				if (num12 < 0.04f)
				{
					num12 = 0.04f;
				}
				float num13 = (float)half_sqrt((float)num12);
				float num14 = expr_2FC.Mass * body.Mass / num12;
				body.ForceTot += num14;
				body.ForceX += num14 * num10 / num13;
				body.ForceY += num14 * num11 / num13;
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	body.Pressure = GAS_K * body.Density;
	if (body.ForceTot > body.Mass * 4.0f & body.Flag == 0)
	{
		body.InRoche = 1;
	}
	else
	{
		if (body.ForceTot * 2.0f < body.Mass * 4.0f)
		{
			body.InRoche = 0;
		}
		else
		{
			if (body.Flag == 2 || body.IsExplosion == 1)
			{
				body.InRoche = 1;
			}
		}
	}
	if (body.Flag == 2)
	{
		body.InRoche = 1;
	}
	outBodies[(a)] = body;
}
// NBodies.Physics.CUDAFloat
int IsNear(struct MeshCell testCell, struct MeshCell neighborCell)
{
	int result = 0;
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			if (neighborCell.IdxX == testCell.IdxX + i && neighborCell.IdxY == testCell.IdxY + j)
			{
				result = 1;
			}
		}
	}
	return result;
}
// NBodies.Physics.CUDAFloat
__kernel  void CalcCollisions(global struct Body* inBodies, int inBodiesLen0, global struct Body* outBodies, int outBodiesLen0, global struct MeshCell* inMesh, int inMeshLen0, global int* meshNeighbors, int meshNeighborsLen0, float dt, float viscosity)
{
	int num = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (num > inBodiesLen0 - 1)
	{
		return;
	}
	struct Body body = inBodies[(num)];
	struct MeshCell meshCell = inMesh[(body.MeshID)];
	for (int i = meshCell.NeighborStartIdx; i < meshCell.NeighborStartIdx + meshCell.NeighborCount; i++)
	{
		int num2 = meshNeighbors[(i)];
		struct MeshCell expr_65 = inMesh[(num2)];
		int bodyStartIdx = expr_65.BodyStartIdx;
		int num3 = expr_65.BodyCount + bodyStartIdx;
		for (int j = bodyStartIdx; j < num3; j++)
		{
			if (j != num)
			{
				struct Body body2 = inBodies[(j)];
				float num4 = body.PosX - body2.PosX;
				float num5 = body.PosY - body2.PosY;
				float num6 = num4 * num4 + num5 * num5;
				float num7 = body.Size * 0.5f + body2.Size * 0.5f;
				if (num6 <= num7 * num7)
				{
					float num8 = (float)half_sqrt((float)num6);
					if (body.InRoche == 1 && body2.InRoche == 1)
					{
						float num9 = 1.192093E-07f;
						float num10 = 3.45267E-11f;
						float num11 = 1.0f;
						if (num6 < num9)
						{
							num8 = num10;
						}
						float num12 = body.Mass * (body.Pressure + body2.Pressure) / (2.0f * body.Density);
						float num13 = -10442.16f * (num11 - num8) * (num11 - num8) / num8;
						float num14 = num4 * num13;
						float num15 = num5 * num13;
						num14 *= num12;
						num15 *= num12;
						body.ForceX -= num14;
						body.ForceY -= num15;
						float num16 = 14.32394f * (num11 - num8);
						float num17 = body2.Mass * num16 * viscosity * 1.0f / body2.Density;
						float num18 = body2.VeloX - body.VeloX;
						float num19 = body2.VeloY - body.VeloY;
						num18 *= num17;
						num19 *= num17;
						body.ForceX += num18;
						body.ForceY += num19;
					}
					else
					{
						if (body.InRoche == 1 && body2.InRoche == 0)
						{
							body.Visible = 0;
						}
						else
						{
							float num20 = (num4 * (body2.VeloX - body.VeloX) + num5 * (body2.VeloY - body.VeloY)) / num6;
							float forceX = num4 * num20;
							float forceY = num5 * num20;
							float colMass = body2.Mass / (body2.Mass + body.Mass);
							if (body.Mass > body2.Mass)
							{
								body = CollideBodies(body, body2, colMass, forceX, forceY);
							}
							else
							{
								if (body.Mass < body2.Mass)
								{
									body.Visible = 0;
								}
								else
								{
									if (body.Mass == body2.Mass)
									{
										if (body.UID > body2.UID)
										{
											body = CollideBodies(body, body2, colMass, forceX, forceY);
										}
										else
										{
											body.Visible = 0;
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	body.VeloX += dt * body.ForceX / body.Mass;
	body.VeloY += dt * body.ForceY / body.Mass;
	body.PosX += dt * body.VeloX;
	body.PosY += dt * body.VeloY;
	if (body.Lifetime > 0.0f)
	{
		body.Age += dt * 4.0f;
	}
	outBodies[(num)] = body;
}
// NBodies.Physics.CUDAFloat
struct Body CollideBodies(struct Body master, struct Body slave, float colMass, float forceX, float forceY)
{
	struct Body body = master;
	body.VeloX += colMass * forceX;
	body.VeloY += colMass * forceY;
	if (body.Flag != 1)
	{
		float arg_70_0 = 3.141593f * (float)pow((double)(body.Size * 0.5f), 2.0);
		float num = 3.141593f * (float)pow((double)(slave.Size * 0.5f), 2.0);
		float num2 = arg_70_0 + num;
		body.Size = (float)half_sqrt((float)((float)((double)num2 / 3.14159265358979))) * 2.0f;
	}
	body.Mass += slave.Mass;
	return body;
}