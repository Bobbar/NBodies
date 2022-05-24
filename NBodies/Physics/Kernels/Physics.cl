#include "Global.h"
#include "Sorting.cl"
#include "Mesh.cl"
#include "Helpers.cl"


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
			float distZ = bodyA.PosZ - bodyB.PosZ;

			float dist = (distX * distX) + (distY * distY) + (distZ * distZ);
			float colDist = aRad + bRad;

			if (dist <= (colDist * colDist))
			{
				float distSqrt = (float)native_sqrt(dist);
				float overlap = 0.5f * (distSqrt - aRad - bRad);

				bodyA.PosX -= overlap * (distX) / distSqrt;
				bodyA.PosY -= overlap * (distY) / distSqrt;
				bodyA.PosZ -= overlap * (distZ) / distSqrt;

			}
		}
	}

	outBodies[i] = bodyA;
}


float3 ComputeForce(float3 posA, float3 posB, float massA, float massB)
{
	//	float3 dir = posA - posB;
	//
	//#if FASTMATH
	//	float dist = dot(dir, dir);
	//	float distSqrt = SQRT(dist);
	//#else
	//	float distSqrt = LENGTH(dir);
	//	float dist = pow(distSqrt, 2);
	//#endif
	//
	//	// Clamp to soften length.
	//	dist = max(dist, SOFTENING);
	//	distSqrt = max(distSqrt, SOFTENING_SQRT);
	//
	//#if FASTMATH
	//	// This math is measurably faster, but its accuracy is questionable,
	//	// so we'll include it in the fast math preprocessor flag.
	//	float force = (massA * massB) / distSqrt / distSqrt / distSqrt;
	//	float3 ret = dir * force;
	//#else
	//	float force = massA * massB / dist;
	//	float3 ret = (dir * force) / distSqrt;
	//#endif
	//
	//	return ret;


	float3 dir = posA - posB;
	float dist = dot(dir, dir);
	float distSqrt = SQRT(dist);

	// Clamp to soften length.
	dist = max(dist, SOFTENING);
	distSqrt = max(distSqrt, SOFTENING_SQRT);

	float force = massA * massB / dist;
	float3 ret = (dir * force) / distSqrt;

	return ret;
}


__kernel void CalcForce(global Body* inBodies, int inBodiesLen, global int4* meshIdxs, global int2* meshNBounds, global int2* meshBodyBounds, global int2* meshChildBounds, global float4* meshCMM, global int2* meshSPL, global int* meshNeighbors, const SimSettings sim, const SPHPreCalc sph, int meshTopStart, int meshTopEnd, global int* postNeeded)
{
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	// Copy body pos & mass.
	float3 iPos = (float3)(inBodies[(a)].PosX, inBodies[(a)].PosY, inBodies[(a)].PosZ);
	float iMass = inBodies[(a)].Mass;

	// Copy body mesh cell.
	int bodyMeshId = inBodies[(a)].MeshID;
	int bodyCellParentID = meshSPL[bodyMeshId].y;
	int4 bodyCellIdx = meshIdxs[bodyMeshId];

	float3 iForce = (float3)(0.0f, 0.0f, 0.0f);
	float iDensity = 0.0f;

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
				int4 childIdx = meshIdxs[c];

				// If the cell is far, compute force from cell.
				// [ Particle -> Mesh ]
				bool far = IsFar(bodyCellIdx, childIdx);
				if (far)
				{
					float4 cellCMM = meshCMM[c];
					iForce += ComputeForce(cellCMM.xyz, iPos, cellCMM.w, iMass);
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
							float3 jPos = (float3)(inBodies[(mb)].PosX, inBodies[(mb)].PosY, inBodies[(mb)].PosZ);
							float jMass = inBodies[(mb)].Mass;
							float3 dir = jPos - iPos;
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

		bodyCellIdx = meshIdxs[bodyCellParentID];
		bodyCellParentID = meshSPL[bodyCellParentID].y;

		done = (bodyCellParentID == -1);
		bottom = false;
	}

	// *** Particle 2 Mesh ***
	// Accumulate force from remaining distant cells at the top-most level.
	for (int top = meshTopStart; top < meshTopEnd; top++)
	{
		int4 topIdx = meshIdxs[top];

		if (IsFar(bodyCellIdx, topIdx))
		{
			float4 topCMM = meshCMM[top];
			iForce += ComputeForce(topCMM.xyz, iPos, topCMM.w, iMass);
		}
	}

	// Check for the phony roche condition.
	if (fast_length(iForce) > (iMass * 4.0f))
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
	inBodies[(a)].ForceZ = iForce.z;
	inBodies[(a)].Density = iDensity;
}



__kernel void ElasticCollisions(global Body* inBodies, int inBodiesLen, global int2* meshSPL, global int2* meshNBounds, global int2* meshBodyBounds, global int* meshNeighbors, int collisions, global int* postNeeded)
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
		int2 parentSPL = meshSPL[outBody.MeshID];
		int pcellSize = parentSPL.x;
		int parentID = parentSPL.y;

		// Move up through parent cells until we find one
		// whose size is atleast as big as the target body.
		while (pcellSize < outBody.Size)
		{
			parentID = parentSPL.y;

			// Stop if we reach the top-most level.
			if (parentID == -1)
				break;

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
					float distZ = outBody.PosZ - inBody.PosZ;

					float dist = distX * distX + distY * distY + distZ * distZ;
					float distSqrt = (float)SQRT(dist);

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


__kernel void SPHCollisions(global Body* inBodies, int inBodiesLen, global Body* outBodies, global int2* meshNBounds, global int2* meshSPL, global int2* meshChildBounds, global int4* meshIdxs, global int2* meshBodyBounds, global int* meshNeighbors, global float3* centerMass, const SimSettings sim, const SPHPreCalc sph, global int* postNeeded)
{
	// Get index for the current body.
	int a = get_global_id(0);

	if (a >= inBodiesLen)
		return;

	float maxTemp = 10000.0f;//4000.0f
	double tempDelta = 0;
	int neighbors = 0;

	// Copy current body from memory.
	Body outBody = inBodies[(a)];
	float3 outPos = (float3)(outBody.PosX, outBody.PosY, outBody.PosZ);

	int bodyCellParentID = meshSPL[outBody.MeshID].y;
	int4 bodyCellIdx = meshIdxs[outBody.MeshID];

	if (sim.CollisionsOn == 1 && HasFlagB(outBody, INROCHE))
	{
		// Iterate parent cell neighbors.
		int2 parentNBounds = meshNBounds[bodyCellParentID];

		int start = parentNBounds.x;
		int len = start + parentNBounds.y;

		// PERF HACK: Mask out the len for bodies at resting density to skip the tree walk?
		len = len * !((outBody.Mass * sph.fDensity) == outBody.Density);

		// Compute pressure from density.
		float oPress = sim.GasK * outBody.Density;

		for (int nc = start; nc < len; nc++)
		{
			// Iterate neighbor child cells.
			int nId = meshNeighbors[(nc)];

			int2 childBounds = meshChildBounds[nId];
			int childStartIdx = childBounds.x;
			int childLen = childStartIdx + childBounds.y;

			for (int c = childStartIdx; c < childLen; c++)
			{
				int4 cellIdx = meshIdxs[c];

				// Check for close cell.
				if (!IsFar(bodyCellIdx, cellIdx))
				{
					// Record # of neighbors encountered.
					neighbors++;

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
							float3 inPos = (float3)(inBody.PosX, inBody.PosY, inBody.PosZ);
							float3 dir = outPos - inPos;

#if FASTMATH
							float dist = dot(dir, dir);
							float distSqrt = SQRT(dist);
#else
							float distSqrt = LENGTH(dir);
							float dist = pow(distSqrt, 2);
#endif

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
								float iPress = sim.GasK * inBody.Density; // Compute pressure from density.
								float pressScalar = outBody.Mass * (oPress + iPress) / (2.0f * outBody.Density);
								float pressGrad = sph.fPressure * kDiff * kDiff / distSqrt;

								outBody.ForceX += (dir.x * pressGrad) * pressScalar;
								outBody.ForceY += (dir.y * pressGrad) * pressScalar;
								outBody.ForceZ += (dir.z * pressGrad) * pressScalar;

								// Viscosity force
								float viscLaplace = sph.fViscosity * kDiff;
								float viscScalar = inBody.Mass * viscLaplace * sim.Viscosity / inBody.Density;

								float veloDiffX = inBody.VeloX - outBody.VeloX;
								float veloDiffY = inBody.VeloY - outBody.VeloY;
								float veloDiffZ = inBody.VeloZ - outBody.VeloZ;

								outBody.ForceX += veloDiffX * viscScalar;
								outBody.ForceY += veloDiffY * viscScalar;
								outBody.ForceZ += veloDiffZ * viscScalar;

								float velo = veloDiffX * veloDiffX + veloDiffY * veloDiffY + veloDiffZ * veloDiffZ;

								// Temp delta from p2p conduction.
								float tempK = 0.5f;//0.5f;//2.0f;//1.0f;
								float tempDiff = outBody.Temp - inBody.Temp;
								tempDelta += (-0.5 * tempK) * (tempDiff / dist);

								// Temp delta from p2p friction.
								float coeff = 0.0005f;//0.0004f;
								float adhesion = 0.1f; //viscosity;//0.1f;
								float heatJ = 8.31f;
								float heatFac = 1950;
								float factor = 0.0975f;
								tempDelta += (factor * velo) / heatJ;

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
	outBody.VeloZ += sim.DeltaTime * outBody.ForceZ / outBody.Mass;

	outBody.PosX += sim.DeltaTime * outBody.VeloX;
	outBody.PosY += sim.DeltaTime * outBody.VeloY;
	outBody.PosZ += sim.DeltaTime * outBody.VeloZ;

	outBody.Temp += tempDelta * sim.DeltaTime;

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
	float dist = DISTANCE(cm, (float3)(outBody.PosX, outBody.PosY, outBody.PosZ));

	if (dist > sim.CullDistance)
	{
		outBody.Flag = SetFlag(outBody.Flag, CULLED, true);
		postNeeded[0] = 1;
	}


	// Black body radiation.
	//double SBC = 0.000000056703;
	double SBC = 0.056703;

	// Only apply black body radiation if we have a temp, and we are not inside a clump. ( neighbors < 27 )
	if (outBody.Temp > 0 && neighbors < 27)
	{
		//double lossD = SBC * pow((double)outBody.Temp, 4.0) * 0.5f;//0.1;//1.0;//0.785;
		double lossD = SBC * pow((double)outBody.Temp, 2.0) * 0.05f;//0.1;//1.0;//0.785;
		outBody.Temp -= ((lossD / (double)(outBody.Mass * 0.25)) / 4.186) * sim.DeltaTime;
	}

	outBody.Temp = min(outBody.Temp, maxTemp);
	outBody.Temp = max(outBody.Temp, 1.0f);


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
