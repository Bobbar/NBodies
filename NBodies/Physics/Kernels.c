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
	float num = 0.3f;
	float num2 = 1.192093E-07f;
	int num3 = get_local_size(0) * get_group_id(0) + get_local_id(0);
	bool flag = num3 > inBodiesLen0 - 1;
	if (!flag)
	{
		struct Body body = inBodies[(num3)];
		struct MeshCell meshCell = inMesh[(body.MeshID)];
		struct MeshCell testCell = meshCell;
		struct MeshCell meshCell2 = inMesh[(meshCell.ParentID)];
		body.ForceTot = 0.0f;
		body.ForceX = 0.0f;
		body.ForceY = 0.0f;
		body.Density = 0.0f;
		body.Pressure = 0.0f;
		float num4 = 1.0f;
		float num5 = 1.0f;
		float num6 = 1.566682f;
		body.Density = body.Mass * num6;
		for (int i = 0; i < topLevel; i++)
		{
			int neighborStartIdx = meshCell2.NeighborStartIdx;
			int num7 = neighborStartIdx + meshCell2.NeighborCount;
			for (int j = neighborStartIdx; j < num7; j++)
			{
				int num8 = meshNeighbors[(j)];
				struct MeshCell meshCell3 = inMesh[(num8)];
				int childStartIdx = meshCell3.ChildStartIdx;
				int num9 = childStartIdx + meshCell3.ChildCount;
				for (int k = childStartIdx; k < num9; k++)
				{
					bool flag2 = k != body.MeshID;
					if (flag2)
					{
						struct MeshCell meshCell4 = inMesh[(k)];
						bool flag3 = IsNear(testCell, meshCell4) == 0;
						if (flag3)
						{
							float num10 = meshCell4.CmX - body.PosX;
							float num11 = meshCell4.CmY - body.PosY;
							float num12 = num10 * num10 + num11 * num11;
							float num13 = (float)half_sqrt((float)num12);
							float num14 = (float)meshCell4.Mass * body.Mass;
							float num15 = num14 / num12;
							body.ForceTot += num15;
							body.ForceX += num15 * num10 / num13;
							body.ForceY += num15 * num11 / num13;
						}
					}
				}
			}
			testCell = meshCell2;
			meshCell2 = inMesh[(meshCell2.ParentID)];
		}
		for (int l = levelIdx[(topLevel)]; l < inMeshLen0; l++)
		{
			struct MeshCell meshCell5 = inMesh[(l)];
			bool flag4 = IsNear(testCell, meshCell5) == 0;
			if (flag4)
			{
				float num10 = meshCell5.CmX - body.PosX;
				float num11 = meshCell5.CmY - body.PosY;
				float num12 = num10 * num10 + num11 * num11;
				float num13 = (float)half_sqrt((float)num12);
				float num14 = (float)meshCell5.Mass * body.Mass;
				float num15 = num14 / num12;
				body.ForceTot += num15;
				body.ForceX += num15 * num10 / num13;
				body.ForceY += num15 * num11 / num13;
			}
		}
		for (int m = meshCell.NeighborStartIdx; m < meshCell.NeighborStartIdx + meshCell.NeighborCount; m++)
		{
			int num16 = meshNeighbors[(m)];
			struct MeshCell meshCell6 = inMesh[(num16)];
			int bodyStartIdx = meshCell6.BodyStartIdx;
			int num17 = meshCell6.BodyCount + bodyStartIdx;
			for (int n = bodyStartIdx; n < num17; n++)
			{
				bool flag5 = n != num3;
				if (flag5)
				{
					struct Body body2 = inBodies[(n)];
					float num10 = body2.PosX - body.PosX;
					float num11 = body2.PosY - body.PosY;
					float num12 = num10 * num10 + num11 * num11;
					bool flag6 = num12 <= num4;
					if (flag6)
					{
						bool flag7 = num12 < num2;
						if (flag7)
						{
							num12 = num2;
						}
						float num18 = num5 - num12;
						float num19 = num6 * num18 * num18 * num18;
						body.Density += body.Mass * num19;
					}
					bool flag8 = num12 < 0.04f;
					if (flag8)
					{
						num12 = 0.04f;
					}
					float num13 = (float)half_sqrt((float)num12);
					float num14 = body2.Mass * body.Mass;
					float num15 = num14 / num12;
					body.ForceTot += num15;
					body.ForceX += num15 * num10 / num13;
					body.ForceY += num15 * num11 / num13;
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		body.Pressure = num * body.Density;
		bool flag9 = body.ForceTot > body.Mass * 4.0f & body.Flag == 0;
		if (flag9)
		{
			body.InRoche = 1;
		}
		else
		{
			bool flag10 = body.ForceTot * 2.0f < body.Mass * 4.0f;
			if (flag10)
			{
				body.InRoche = 0;
			}
			else
			{
				bool flag11 = body.Flag == 2 || body.IsExplosion == 1;
				if (flag11)
				{
					body.InRoche = 1;
				}
			}
		}
		bool flag12 = body.Flag == 2;
		if (flag12)
		{
			body.InRoche = 1;
		}
		outBodies[(num3)] = body;
	}
}
// NBodies.Physics.CUDAFloat
  int IsNear(struct MeshCell testCell, struct MeshCell neighborCell)
{
	int result = 0;
	for (int num = -1; num <= 1; num++)
	{
		for (int num2 = -1; num2 <= 1; num2++)
		{
			bool flag = neighborCell.IdxX == testCell.IdxX + num && neighborCell.IdxY == testCell.IdxY + num2;
			if (flag)
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
	bool flag = num > inBodiesLen0 - 1;
	if (!flag)
	{
		struct Body body = inBodies[(num)];
		struct MeshCell meshCell = inMesh[(body.MeshID)];
		for (int i = meshCell.NeighborStartIdx; i < meshCell.NeighborStartIdx + meshCell.NeighborCount; i++)
		{
			int num2 = meshNeighbors[(i)];
			struct MeshCell meshCell2 = inMesh[(num2)];
			int bodyStartIdx = meshCell2.BodyStartIdx;
			int num3 = meshCell2.BodyCount + bodyStartIdx;
			for (int j = bodyStartIdx; j < num3; j++)
			{
				bool flag2 = j != num;
				if (flag2)
				{
					struct Body body2 = inBodies[(j)];
					float num4 = body.PosX - body2.PosX;
					float num5 = body.PosY - body2.PosY;
					float num6 = num4 * num4 + num5 * num5;
					float num7 = body.Size * 0.5f + body2.Size * 0.5f;
					bool flag3 = num6 <= num7 * num7;
					if (flag3)
					{
						float num8 = (float)half_sqrt((float)num6);
						bool flag4 = body.InRoche == 1 && body2.InRoche == 1;
						if (flag4)
						{
							float num9 = 1.192093E-07f;
							float num10 = 3.45267E-11f;
							float num11 = 1.0f;
							bool flag5 = num6 < num9;
							if (flag5)
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
							bool flag6 = body.InRoche == 1 && body2.InRoche == 0;
							if (flag6)
							{
								body.Visible = 0;
							}
							else
							{
								float num20 = num4 * (body2.VeloX - body.VeloX) + num5 * (body2.VeloY - body.VeloY);
								float num21 = num20 / num6;
								float forceX = num4 * num21;
								float forceY = num5 * num21;
								float colMass = body2.Mass / (body2.Mass + body.Mass);
								bool flag7 = body.Mass > body2.Mass;
								if (flag7)
								{
									body = CollideBodies(body, body2, colMass, forceX, forceY);
								}
								else
								{
									bool flag8 = body.Mass < body2.Mass;
									if (flag8)
									{
										body.Visible = 0;
									}
									else
									{
										bool flag9 = body.Mass == body2.Mass;
										if (flag9)
										{
											bool flag10 = body.UID > body2.UID;
											if (flag10)
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
		bool flag11 = body.Lifetime > 0.0f;
		if (flag11)
		{
			body.Age += dt * 4.0f;
		}
		outBodies[(num)] = body;
	}
}
// NBodies.Physics.CUDAFloat
  struct Body CollideBodies(struct Body master, struct Body slave, float colMass, float forceX, float forceY)
{
	struct Body body = master;
	body.VeloX += colMass * forceX;
	body.VeloY += colMass * forceY;
	bool flag = body.Flag != 1;
	if (flag)
	{
		float num = 3.141593f * (float)pow((double)(body.Size * 0.5f), 2.0);
		float num2 = 3.141593f * (float)pow((double)(slave.Size * 0.5f), 2.0);
		float num3 = num + num2;
		body.Size = (float)half_sqrt((float)((float)((double)num3 / 3.14159265358979))) * 2.0f;
	}
	body.Mass += slave.Mass;
	return body;
}
