#define _USE_MATH_DEFINES

#include <stdio.h>
#include <assert.h>
#include <functional>
#include "MultiScale.h"
#include <time.h>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#define LOG_1_5 0.4054651
#define INCLUDE_FLANN
#ifdef INCLUDE_FLANN
#include "flann/flann.h"
#endif


using namespace std;

void
vec3D_euler_rotation(
				   float *pfx, // 1x3
				   float *pfy, // 1x3
				   float *pfz, // 1x3
				   float *ori // 3x3
				   )
{
	// Better check for degeneracies, gimbal lock

	float fZXYMagSqr = pfz[0]*pfz[0]+pfz[1]*pfz[1];

	float fAlph = atan2( -pfz[1], pfz[0] );
	float fBeta = atan2(  pfz[2], sqrt(fZXYMagSqr) );
	float fGamm = atan2(  pfy[2], pfx[2] );

	float ca = cos(fAlph);
	float sa = sin(fAlph);

	float cb = cos(fBeta);
	float sb = sin(fBeta);

	float cg = cos(fGamm);
	float sg = sin(fGamm);

	ori[3*0+0]= ca*cg-cb*sa*sg;
	ori[3*0+1]= cg*sa-ca*cb*sg;
	ori[3*0+2]= sb*sg;

	ori[3*1+0]=-cb*cg*sa-ca*sg;
	ori[3*1+1]= ca*cb*cg-sa*sg;
	ori[3*1+2]= cg*sb;

	ori[3*2+0]= sa*sb;
	ori[3*2+1]=-ca*sb;
	ori[3*2+2]= cb;
}


int
compatible_features(
					const Feature3DInfo &f1,
					const Feature3DInfo &f2,
					float fScaleDiffThreshold = LOG_1_5,
					float fShiftThreshold = 0.5,
					float fCosineAngleThreshold = -1.0f
					)
{
	if( (f1.m_uiInfo & INFO_FLAG_LINE) != (f2.m_uiInfo & INFO_FLAG_LINE) )
	{
		// Not the same geometry
		return 0;
	}
	else  if(
		(f1.m_uiInfo & INFO_FLAG_LINE) == INFO_FLAG_LINE
		)
	{
		// Lines

		// Euclidean distance of features
		float fdx = f1.x-f2.x;
		float fdy = f1.y-f2.y;
		float fdz = f1.z-f2.z;
		float fDist12_1 = sqrt( fdx*fdx + fdy*fdy + fdz*fdz );

		fdx = f1.ori[0][0] - f2.ori[0][0];
		fdy = f1.ori[0][1] - f2.ori[0][1];
		fdz = f1.ori[0][2] - f2.ori[0][2];
		float fDist12_2 = sqrt( fdx*fdx + fdy*fdy + fdz*fdz );

		fdx = f1.ori[0][0] - f1.x;
		fdy = f1.ori[0][1] - f1.y;
		fdz = f1.ori[0][2] - f1.z;
		float fLength1 = sqrt( fdx*fdx + fdy*fdy + fdz*fdz );

		fdx = f2.ori[0][0] - f2.x;
		fdy = f2.ori[0][1] - f2.y;
		fdz = f2.ori[0][2] - f2.z;
		float fLength2 = sqrt( fdx*fdx + fdy*fdy + fdz*fdz );

		float fMeasure = (fDist12_1+fDist12_2)/(fLength1+fLength2);

		if( fMeasure < fShiftThreshold )
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
	else if(
		(f1.m_uiInfo & INFO_FLAG_LINE) == 0
		)
	{
		// Spheres

		// Euclidean distance of features
		float fdx = f1.x-f2.x;
		float fdy = f1.y-f2.y;
		float fdz = f1.z-f2.z;
		float fDist = sqrt( fdx*fdx + fdy*fdy + fdz*fdz );

		//float fScaleDiffThreshold = (float)log(1.5f);
		//float fShiftThreshold = 0.5; // works the best, MICCAI 2009 results
		//////float fShiftThreshold = 0.75;
		//////float fShiftThreshold = 0.4;
		//////float fShiftThreshold = 0.6;
		//////float fShiftThreshold = 1.0;

		//printf( "Shift threshold: %f\n", fShiftThreshold );

		float fScaleDiff = (float)fabs( log(f1.scale/f2.scale) );

		float fMinCosineAngle = vec3D_dot_3d( &f1.ori[0][0], &f2.ori[0][0] );
		if( vec3D_dot_3d( &f1.ori[1][0], &f2.ori[1][0] ) < fMinCosineAngle ) fMinCosineAngle = vec3D_dot_3d( &f1.ori[1][0], &f2.ori[1][0] );
		if( vec3D_dot_3d( &f1.ori[2][0], &f2.ori[2][0] ) < fMinCosineAngle ) fMinCosineAngle = vec3D_dot_3d( &f1.ori[2][0], &f2.ori[2][0] );

		if( fScaleDiff < fScaleDiffThreshold
			&&
			fDist < fShiftThreshold*f1.scale
			&&
			fCosineAngleThreshold < fMinCosineAngle
			)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		// Shapes incompatible
		return 0;
	}
}

//
// vec3D_euler_rotation()
//
// Calculate euler rotation matrix from an arbitrary basis
//
void
vec3D_euler_rotation(
				   float *pfx0, // 1x3
				   float *pfy0, // 1x3
				   float *pfz0, // 1x3

				   float *pfx1, // 1x3
				   float *pfy1, // 1x3
				   float *pfz1, // 1x3

				   float *ori // 3x3
				   )
{
	// Better check for degeneracies, gimbal lock

	float fz0z1Dot = pfz0[0]*pfz1[0] + pfz0[1]*pfz1[1] + pfz0[2]*pfz1[2];
	float pfz0z1Cross[3];
	if( fz0z1Dot == 1.0 || fz0z1Dot == -1.0 )
	{
		// Invalid
		return;
	}
	vec3D_cross_3d( pfz0, pfz1, pfz0z1Cross );
	float fz0z1CrossMag = vec3D_mag( pfz0z1Cross );
	if( fz0z1CrossMag == 0 )
	{
		// Invalid
		return;
	}

	float fx0CrossDot = vec3D_dot_3d(pfx0,pfz0z1Cross);
	float fy0CrossDot = vec3D_dot_3d(pfy0,pfz0z1Cross);

	float fx1CrossDot = vec3D_dot_3d(pfx1,pfz0z1Cross);
	float fy1CrossDot = vec3D_dot_3d(pfy1,pfz0z1Cross);

	float fAlph = atan2(  fx0CrossDot, fy0CrossDot );
	float fBeta = atan2(     fz0z1Dot, fz0z1CrossMag );
	float fGamm = -atan2( fx1CrossDot, fy1CrossDot );

	float ca = cos(fAlph);
	float sa = sin(fAlph);

	float cb = cos(fBeta);
	float sb = sin(fBeta);

	float cg = cos(fGamm);
	float sg = sin(fGamm);

	ori[3*0+0]= ca*cg-cb*sa*sg;
	//ori[3*0+1]= cg*sa-ca*cb*sg;
	ori[3*0+1]= cg*sa+ca*cb*sg;
	ori[3*0+2]= sb*sg;

	ori[3*1+0]=-cb*cg*sa-ca*sg;
	ori[3*1+1]= ca*cb*cg-sa*sg;
	ori[3*1+2]= cg*sb;

	ori[3*2+0]= sa*sb;
	ori[3*2+1]=-ca*sb;
	ori[3*2+2]= cb;

	// Another definition

	ori[3*0+0]= ca*cg-cb*sa*sg;
	//ori[3*0+1]= cg*sa-ca*cb*sg;
	ori[3*0+1]= cg*sa+ca*cb*sg;
	ori[3*0+2]= sb*sg;

	ori[3*1+0]=-cb*cg*sa-ca*sg;
	ori[3*1+1]= ca*cb*cg-sa*sg;
	ori[3*1+2]= cg*sb;

	ori[3*2+0]= sa*sb;
	ori[3*2+1]=-ca*sb;
	ori[3*2+2]= cb;
}

int
determine_rotation_3point(
					   float *pfP01, float *pfP02, float *pfP03, // points in image 1 (3d)
					   float *rot // rotation matrix  (3x3)
					   )
{
	float pfV0_12[3];
	float pfV0_13[3];
	float pfV0_nm[3];

	// Subtract point 1 to convert to vectors
	vec3D_diff_3d( pfP01, pfP02, pfV0_12 );
	vec3D_diff_3d( pfP01, pfP03, pfV0_13 );

	// Normalize vectors
	vec3D_norm_3d( pfV0_12 );
	vec3D_norm_3d( pfV0_13 );

	// Cross product between 2 vectors to get normal
	vec3D_cross_3d( pfV0_12, pfV0_13, pfV0_nm );
	vec3D_norm_3d(  pfV0_nm );

	// Cross product between 1st vectors and normal to get orthogonal 3rd vector
	vec3D_cross_3d( pfV0_nm, pfV0_12, pfV0_13 );
	vec3D_norm_3d(  pfV0_13 );

	rot[0]=pfV0_12[0];
	rot[1]=pfV0_12[1];
	rot[2]=pfV0_12[2];

	rot[3]=pfV0_13[0];
	rot[4]=pfV0_13[1];
	rot[5]=pfV0_13[2];

	rot[6]=pfV0_nm[0];
	rot[7]=pfV0_nm[1];
	rot[8]=pfV0_nm[2];

	return 1;
}
//
// determine_rotation_3pt()
//
// Determine rotation matrix between 3 pairs of corresponding points.
//
int
determine_rotation_3point(
  float *pfP01, float *pfP02, float *pfP03,
  float *pfP11, float *pfP12, float *pfP13,
  float *rot
  )
{
float pfV0_12[3];
float pfV0_13[3];
float pfV0_nm[3];

float pfV1_12[3];
float pfV1_13[3];
float pfV1_nm[3];

// Subtract point 1 to convert to vectors
vec3D_diff_3d( pfP01, pfP02, pfV0_12 );
vec3D_diff_3d( pfP01, pfP03, pfV0_13 );
vec3D_diff_3d( pfP11, pfP12, pfV1_12 );
vec3D_diff_3d( pfP11, pfP13, pfV1_13 );

// Normalize vectors
vec3D_norm_3d( pfV0_12 );
vec3D_norm_3d( pfV0_13 );
vec3D_norm_3d( pfV1_12 );
vec3D_norm_3d( pfV1_13 );

// Cross product between 2 vectors to get normal
vec3D_cross_3d( pfV0_12, pfV0_13, pfV0_nm );
vec3D_cross_3d( pfV1_12, pfV1_13, pfV1_nm );
vec3D_norm_3d(  pfV0_nm );
vec3D_norm_3d(  pfV1_nm );

// Cross product between 1st vectors and normal to get orthogonal 3rd vector
vec3D_cross_3d( pfV0_nm, pfV0_12, pfV0_13 );
vec3D_cross_3d( pfV1_nm, pfV1_12, pfV1_13 );
vec3D_norm_3d(  pfV0_13 );
vec3D_norm_3d(  pfV1_13 );

// Generate euler rotation
vec3D_euler_rotation(
pfV0_12, pfV0_13, pfV0_nm,
pfV1_12, pfV1_13, pfV1_nm,
rot );

return 1;
}

int
msComputeNearestNeighborDistanceRatioInfo(
					 vector<Feature3DInfo> &vecFeats1,
					 vector<Feature3DInfo> &vecFeats2,
					 vector<int>		&vecMatchIndex12,
					 vector<float>		&vecMatchDist12
						  )
{
	int iCompatible = 0;

	int iPCs = 64;

	for( int i = 0; i < vecFeats1.size(); i++ )
	{
		float fMinDist = 0;//vecFeats1[i].DistSqrPCs( vecFeats2[0], iPCs );
		int iMinDistIndex = 0;

		float fMinDist2 = 0;//vecFeats1[i].DistSqrPCs( vecFeats2[1], iPCs );
		int iMinDist2Index = 1;
		if( fMinDist2 < fMinDist )
		{
			float fTemp = fMinDist;
			fMinDist = fMinDist2;
			fMinDist2 = fTemp;
			iMinDistIndex = 1;
			iMinDist2Index = 0;
		}

		for( int j = 2; j < vecFeats2.size(); j++ )
		{
			float fDist = 0;//vecFeats1[i].DistSqrPCs( vecFeats2[j], iPCs );
			if( fDist < fMinDist2 )
			{
				// 1st and 2nd matches shoud not be compatible

				if( fDist < fMinDist )
				{
					if( !compatible_features( vecFeats2[j], vecFeats2[iMinDistIndex] ) ) // Add this line for dense features
					{
						// Closest feature is not geometrically compatible with first
						// Shuffle down 1st and 2nd nearest distances
						fMinDist2 = fMinDist;
						iMinDist2Index = iMinDistIndex;
						fMinDist = fDist;
						iMinDistIndex = j;
					}
					else
					{
						// Closest feature is geometrically compatible with first
						// Replace only 1st feature, we have found a better instance of the same thing
						fMinDist = fDist;
						iMinDistIndex = j;
					}
				}
				else
				{
					if( !compatible_features( vecFeats2[j], vecFeats2[iMinDistIndex] ) ) // Add this line for dense features
					{
						// Second closest feature is not geometrically compatible with first
						// Shuffle down 2nd nearest distance
						fMinDist2 = fDist;
						iMinDist2Index = j;
					}
					else
					{
						//printf( "worse \n" );
						// Second closest feature is geometrically compatible with first
						// It is a worse instance of the current closest feature, ignore it
					}
				}
			}
		}

		if( compatible_features( vecFeats2[iMinDist2Index], vecFeats2[iMinDistIndex] ) )
		{
			iCompatible++;
			//fMinDist = 1;
			//fMinDist2 = 10000;
		}

		vecMatchIndex12.push_back( iMinDistIndex );
		//vecMatchDist12.push_back( fMinDist );
		vecMatchDist12.push_back( fMinDist/fMinDist2 );
	}
	return 0;
}

bool pairIntFloatLeastToGreatest( const pair<int,float> &ii1, const pair<int,float> &ii2 )
{
	return ii1.second < ii2.second;
}

int
getMinMaxDim(
			 vector<Feature3DInfo> &vecImgFeats2,
			 float *pfMin,
			 float *pfMax
			 )
{
	pfMin[0] = pfMax[0] = vecImgFeats2[0].x;
	pfMin[1] = pfMax[1] = vecImgFeats2[0].y;
	pfMin[2] = pfMax[2] = vecImgFeats2[0].z;
	for( int i = 0; i < vecImgFeats2.size(); i++ )
	{
		if( vecImgFeats2[i].x > pfMax[0] ) pfMax[0] = vecImgFeats2[i].x;
		if( vecImgFeats2[i].x < pfMin[0] ) pfMin[0] = vecImgFeats2[i].x;

		if( vecImgFeats2[i].y > pfMax[1] ) pfMax[1] = vecImgFeats2[i].y;
		if( vecImgFeats2[i].y < pfMin[1] ) pfMin[1] = vecImgFeats2[i].y;

		if( vecImgFeats2[i].z > pfMax[2] ) pfMax[2] = vecImgFeats2[i].z;
		if( vecImgFeats2[i].z < pfMin[2] ) pfMin[2] = vecImgFeats2[i].z;
	}
	return 0;
}
/*
//
// vec3D_euler_rotation()
//
// Calculate euler rotation matrix from standard reference space
// (100), (010), (001)
// to an arbitrary.
//
void
vec3D_euler_rotation(
				   float *pfx, // 1x3
				   float *pfy, // 1x3
				   float *pfz, // 1x3
				   float *ori // 3x3
				   )
{
	// Better check for degeneracies, gimbal lock

	float fZXYMagSqr = pfz[0]*pfz[0]+pfz[1]*pfz[1];

	float fAlph = atan2( -pfz[1], pfz[0] );
	float fBeta = atan2(  pfz[2], sqrt(fZXYMagSqr) );
	float fGamm = atan2(  pfy[2], pfx[2] );

	float ca = cos(fAlph);
	float sa = sin(fAlph);

	float cb = cos(fBeta);
	float sb = sin(fBeta);

	float cg = cos(fGamm);
	float sg = sin(fGamm);

	ori[3*0+0]= ca*cg-cb*sa*sg;
	ori[3*0+1]= cg*sa-ca*cb*sg;
	ori[3*0+2]= sb*sg;

	ori[3*1+0]=-cb*cg*sa-ca*sg;
	ori[3*1+1]= ca*cb*cg-sa*sg;
	ori[3*1+2]= cg*sb;

	ori[3*2+0]= sa*sb;
	ori[3*2+1]=-ca*sb;
	ori[3*2+2]= cb;
}

int
compatible_features(
					const Feature3DInfo &f1,
					const Feature3DInfo &f2,
					float fScaleDiffThreshold,
					float fShiftThreshold,
					float fCosineAngleThreshold
					)
{
	if( (f1.m_uiInfo & INFO_FLAG_LINE) != (f2.m_uiInfo & INFO_FLAG_LINE) )
	{
		// Not the same geometry
		return 0;
	}
	else  if(
		(f1.m_uiInfo & INFO_FLAG_LINE) == INFO_FLAG_LINE
		)
	{
		// Lines

		// Euclidean distance of features
		float fdx = f1.x-f2.x;
		float fdy = f1.y-f2.y;
		float fdz = f1.z-f2.z;
		float fDist12_1 = sqrt( fdx*fdx + fdy*fdy + fdz*fdz );

		fdx = f1.ori[0][0] - f2.ori[0][0];
		fdy = f1.ori[0][1] - f2.ori[0][1];
		fdz = f1.ori[0][2] - f2.ori[0][2];
		float fDist12_2 = sqrt( fdx*fdx + fdy*fdy + fdz*fdz );

		fdx = f1.ori[0][0] - f1.x;
		fdy = f1.ori[0][1] - f1.y;
		fdz = f1.ori[0][2] - f1.z;
		float fLength1 = sqrt( fdx*fdx + fdy*fdy + fdz*fdz );

		fdx = f2.ori[0][0] - f2.x;
		fdy = f2.ori[0][1] - f2.y;
		fdz = f2.ori[0][2] - f2.z;
		float fLength2 = sqrt( fdx*fdx + fdy*fdy + fdz*fdz );

		float fMeasure = (fDist12_1+fDist12_2)/(fLength1+fLength2);

		if( fMeasure < fShiftThreshold )
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
	else if(
		(f1.m_uiInfo & INFO_FLAG_LINE) == 0
		)
	{
		// Spheres

		// Euclidean distance of features
		float fdx = f1.x-f2.x;
		float fdy = f1.y-f2.y;
		float fdz = f1.z-f2.z;
		float fDist = sqrt( fdx*fdx + fdy*fdy + fdz*fdz );

		//float fScaleDiffThreshold = (float)log(1.5f);
		//float fShiftThreshold = 0.5; // works the best, MICCAI 2009 results
		//////float fShiftThreshold = 0.75;
		//////float fShiftThreshold = 0.4;
		//////float fShiftThreshold = 0.6;
		//////float fShiftThreshold = 1.0;

		//printf( "Shift threshold: %f\n", fShiftThreshold );

		float fScaleDiff = (float)fabs( log(f1.scale/f2.scale) );

		float fMinCosineAngle = vec3D_dot_3d( &f1.ori[0][0], &f2.ori[0][0] );
		if( vec3D_dot_3d( &f1.ori[1][0], &f2.ori[1][0] ) < fMinCosineAngle ) fMinCosineAngle = vec3D_dot_3d( &f1.ori[1][0], &f2.ori[1][0] );
		if( vec3D_dot_3d( &f1.ori[2][0], &f2.ori[2][0] ) < fMinCosineAngle ) fMinCosineAngle = vec3D_dot_3d( &f1.ori[2][0], &f2.ori[2][0] );

		if( fScaleDiff < fScaleDiffThreshold
			&&
			fDist < fShiftThreshold*f1.scale
			&&
			fCosineAngleThreshold < fMinCosineAngle
			)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		// Shapes incompatible
		return 0;
	}
}*/

//
// determine_rotation_3pt()
//
// Determine rotation matrix between 3 pairs of corresponding points.
//
/*
int
determine_rotation_3point(
					   float *pfP01, float *pfP02, float *pfP03,
					   float *pfP11, float *pfP12, float *pfP13,
					   float *rot
					   )
{
	float pfV0_12[3];
	float pfV0_13[3];
	float pfV0_nm[3];

	float pfV1_12[3];
	float pfV1_13[3];
	float pfV1_nm[3];

	// Subtract point 1 to convert to vectors
	vec3D_diff_3d( pfP01, pfP02, pfV0_12 );
	vec3D_diff_3d( pfP01, pfP03, pfV0_13 );
	vec3D_diff_3d( pfP11, pfP12, pfV1_12 );
	vec3D_diff_3d( pfP11, pfP13, pfV1_13 );

	// Normalize vectors
	vec3D_norm_3d( pfV0_12 );
	vec3D_norm_3d( pfV0_13 );
	vec3D_norm_3d( pfV1_12 );
	vec3D_norm_3d( pfV1_13 );

	// Cross product between 2 vectors to get normal
	vec3D_cross_3d( pfV0_12, pfV0_13, pfV0_nm );
	vec3D_cross_3d( pfV1_12, pfV1_13, pfV1_nm );
	vec3D_norm_3d(  pfV0_nm );
	vec3D_norm_3d(  pfV1_nm );

	// Cross product between 1st vectors and normal to get orthogonal 3rd vector
	vec3D_cross_3d( pfV0_nm, pfV0_12, pfV0_13 );
	vec3D_cross_3d( pfV1_nm, pfV1_12, pfV1_13 );
	vec3D_norm_3d(  pfV0_13 );
	vec3D_norm_3d(  pfV1_13 );

	// Generate euler rotation
	vec3D_euler_rotation(
		pfV0_12, pfV0_13, pfV0_nm,
		pfV1_12, pfV1_13, pfV1_nm,
		rot );

	//float ori_here[3][3];
	//memcpy( &(ori_here[0][0]), ori, sizeof(float)*3*3 );

	//// Rotate template
	//for( int zz = 0; zz < fioImg.z; zz++ )
	//{
	//	for( int yy = 0; yy < fioImg.y; yy++ )
	//	{
	//		for( int xx = 0; xx < fioImg.x; xx++ )
	//		{
	//			float pf1[3] = {xx-fRadius,yy-fRadius,zz-fRadius};
	//			float pf2[3];

	//			mult_3x3<float,float>( ori_here, pf1, pf2 );

	//			// Add center offset
	//			pf2[0] += fRadius;
	//			pf2[1] += fRadius;
	//			pf2[2] += fRadius;

	//			float fPix = fioGetPixelTrilinearInterp( fioImg, pf2[0], pf2[1], pf2[2] );
	//			fT0.data_zyx[zz][yy][xx] = fPix;
	//		}
	//	}
	//}
	return 1;
}*/

float
vec3D_dist_3d(
			 float *pv1,
			 float *pv2
			 )
{

	float dx = pv2[0]-pv1[0];
	float dy = pv2[1]-pv1[1];
	float dz = pv2[2]-pv1[2];
	return sqrt( dx*dx+dy*dy+dz*dz );
}

//
// determine_similarity_transform_3point()
//
// Determines a 3 point similarity transform in 3D.
//
// Inputs:
//    Corresponding points
//		float *pfP01, float *pfP02, float *pfP03,
//		float *pfP11, float *pfP12, float *pfP13,
//
// Outputs
//		rot0: rotation matrix, about the first pair of corresponding points pfP01<->pfP11.
//      fScaleDiff: scale change.
//		(rot1: used internally)
//
int
determine_similarity_transform_3point(
					   float *pfP01, float *pfP02, float *pfP03,
					   float *pfP11, float *pfP12, float *pfP13,

					   // Output
					   // Rotation about point pfP01/pfP11
					   float *rot0,
					   float *rot1,

					   // Scale change (magnification) from image 1 to image 2
					   float &fScaleDiff
					   )
{
	// Points cannot be equal or colinear

	float fDist012 = vec3D_dist_3d( pfP01, pfP02 );
	float fDist013 = vec3D_dist_3d( pfP01, pfP03 );
	float fDist023 = vec3D_dist_3d( pfP02, pfP03 );
	float fDist112 = vec3D_dist_3d( pfP11, pfP12 );
	float fDist113 = vec3D_dist_3d( pfP11, pfP13 );
	float fDist123 = vec3D_dist_3d( pfP12, pfP13 );

	if(
		fDist012 == 0 ||
		fDist013 == 0 ||
		fDist023 == 0 ||
		fDist112 == 0 ||
		fDist113 == 0 ||
		fDist123 == 0 )
	{
		// Points have to be distinct
		return -1;
	}

	// Scale change calculated from correspondences 1 & 2
	// Calculate geometric mean
	//fScaleDiff =
	//	log( fDist112/fDist012 ) +
	//	log( fDist113/fDist013 ) +
	//	log( fDist123/fDist023 );
	//fScaleDiff /= 3.0f;
	//fScaleDiff = exp( fScaleDiff );

//	fScaleDiff = fDist112/fDist012;
	fScaleDiff = (fDist112+fDist113+fDist123)/(fDist012+fDist013+fDist023);

	// Determine rotation about point pfP01/pfP11

	int iReturn;

	iReturn = determine_rotation_3point( pfP01, pfP02, pfP03, rot0 );
	iReturn = determine_rotation_3point( pfP11, pfP12, pfP13, rot1 );

	// Create
	float ori0[3][3];
	memcpy( &(ori0[0][0]), rot0, sizeof(float)*3*3 );
	float ori1[3][3];
	memcpy( &(ori1[0][0]), rot1, sizeof(float)*3*3 );

	float ori1trans[3][3];
	transpose_3x3<float,float>( ori1, ori1trans );

	float rot_one[3][3];
	mult_3x3_matrix<float,float>( ori1trans, ori0, rot_one );

	memcpy( rot0, &(rot_one[0][0]), sizeof(float)*3*3 );

	return iReturn;
}


int
feature_to_three_points(
									  float *pfPoint,
									  float *pfOri, // 3x3 Orientation
									  float fScale,
									  float *pfPoints // 3 output points
								   )
{
	//
	// Nasty bug - took 1 1/2 days to figure out!!!
	//
	//pfPoints[0*3+0] = pfPoint[0] + fScale*pfOri[0*3+0];
	//pfPoints[0*3+1] = pfPoint[1] + fScale*pfOri[1*3+0];
	//pfPoints[0*3+2] = pfPoint[2] + fScale*pfOri[2*3+0];
	//
	//pfPoints[1*3+0] = pfPoint[0] + fScale*pfOri[0*3+1];
	//pfPoints[1*3+1] = pfPoint[1] + fScale*pfOri[1*3+1];
	//pfPoints[1*3+2] = pfPoint[2] + fScale*pfOri[2*3+1];
	//
	//pfPoints[2*3+0] = pfPoint[0] + fScale*pfOri[0*3+2];
	//pfPoints[2*3+1] = pfPoint[1] + fScale*pfOri[1*3+2];
	//pfPoints[2*3+2] = pfPoint[2] + fScale*pfOri[2*3+2];

	pfPoints[0*3+0] = pfPoint[0] + fScale*pfOri[0*3+0];
	pfPoints[0*3+1] = pfPoint[1] + fScale*pfOri[0*3+1];
	pfPoints[0*3+2] = pfPoint[2] + fScale*pfOri[0*3+2];

	pfPoints[1*3+0] = pfPoint[0] + fScale*pfOri[1*3+0];
	pfPoints[1*3+1] = pfPoint[1] + fScale*pfOri[1*3+1];
	pfPoints[1*3+2] = pfPoint[2] + fScale*pfOri[1*3+2];

	pfPoints[2*3+0] = pfPoint[0] + fScale*pfOri[2*3+0];
	pfPoints[2*3+1] = pfPoint[1] + fScale*pfOri[2*3+1];
	pfPoints[2*3+2] = pfPoint[2] + fScale*pfOri[2*3+2];



	return 1;
}

int
determine_similarity_transform_hough(
									  float *pf0, // Points in image 1
									  float *pf1, // Points in image 2
									  float *pfs0, // Scale for points in image 1
									  float *pfs1, // Scale for points in image 2
									  float *pfo0, // 3x3 Orientation for points in image 1
									  float *pfo1, // 3x3 Orientation for points in image 2
									  float *pfProb, // Probability associated with match

									  int iPoints, // number of points
									  int iIterations, // number of iterations

									  float *pfC0, // Reference center point in image 1

									  // Output

									  // Reference center in image 2
									  float *pfC1,

									  // Rotation about point pfP01/pfP11
									  float *rot,

									  // Scale change (magnification) from image 1 to image 2
									  float &fScaleDiff,

									  // Flag inliers
									  int *piInliers
									  )
{
	//ScaleSpaceXYZHash ssh;

	int iMaxInliers = -1;
	int iMaxInlier1 = -1;
	int iMaxInlier2 = -1;
	int iMaxInlier3 = -1;

	float fMaxInlierProb = 0;

	// Go through entire set of points
	for( int i = 0; i < iPoints; i++ )
	{
		float pts0[3*3];
		float pts1[3*3];

		feature_to_three_points( pf0 + i*3, pfo0 + i*3*3, pfs0[i], pts0 );
		feature_to_three_points( pf1 + i*3, pfo1 + i*3*3, pfs1[i], pts1 );

		// Calculate transform
		float rot_here0[3][3];
		float rot_here1[3][3];
		float fScaleDiffHere;
		determine_similarity_transform_3point(
					   pts0+0*3, pts0+1*3, pts0+2*3,
					   pts1+0*3, pts1+1*3, pts1+2*3,
					   (float*)(&(rot_here0[0][0])),
					   (float*)(&(rot_here1[0][0])),
					   fScaleDiffHere
					   );

		//fScaleDiffHere = 1.0f / fScaleDiffHere;

		// Count the number of inliers
		int iInliers = 0;
		float fInlierProb = 0;
		float pfTest[3];
		for( int j = 0; j < iPoints; j++ )
		{
			// Transform point j in image 1
			similarity_transform_3point(
				pf0+j*3, pfTest,
				pf0+i*3, pf1+i*3,
				(float*)(&(rot_here0[0][0])), fScaleDiffHere );

			//float pfStuff[3] = {0,0,0};
			//float pfCenter2[3] = {0,0,0};
			//similarity_transform_3point(
			//	pfStuff, pfCenter2,
			//	pf0+i*3, pf1+i*3,
			//	(float*)(&(rot_here0[0][0])), fScaleDiffHere );

			//similarity_transform_3point(
			//	pf0+j*3, pfTest,
			//	pfStuff, pfCenter2,
			//	(float*)(&(rot_here0[0][0])), fScaleDiffHere );


			// Evaluate
			Feature3DInfo f1Real;
			Feature3DInfo f1Test;
			f1Real.x = pf1[j*3+0];
			f1Real.y = pf1[j*3+1];
			f1Real.z = pf1[j*3+2];
			f1Real.scale = pfs1[j];

			f1Test.x = pfTest[0];
			f1Test.y = pfTest[1];
			f1Test.z = pfTest[2];
			f1Test.scale = pfs0[j]*fScaleDiffHere;

// These parameters tuned from M. Styners Krabbe disease dataset
// Note these parameters quantify displacements of individual features, not transform
#define HOUGH_THRES_SCALE 1.0
#define HOUGH_THRES_TRANS 2.0
#define HOUGH_THRES_ORIEN 0.7
//#define HOUGH_THRES_SCALE 0.5
//#define HOUGH_THRES_TRANS 2.0
//#define HOUGH_THRES_ORIEN 0.7

			if( compatible_features( f1Real, f1Test, HOUGH_THRES_SCALE, HOUGH_THRES_TRANS ) )
			{
				float rot_temp[3][3];
				memcpy( &(f1Real.ori[0][0]), pfo1 + j*3*3, 3*3*sizeof(float) );
				memcpy( &(rot_temp[0][0]), pfo0 + j*3*3, 3*3*sizeof(float) );
				transpose_3x3<float,float>( rot_temp, f1Test.ori );
				mult_3x3_matrix<float,float>( rot_here0, f1Test.ori, rot_temp );
				transpose_3x3<float,float>( rot_temp, f1Test.ori );
				if( compatible_features( f1Real, f1Test, HOUGH_THRES_SCALE, HOUGH_THRES_TRANS, HOUGH_THRES_ORIEN ) )
				{
					iInliers++;
					fInlierProb += pfProb[j];
				}
			}
		}
		//if( iInliers > iMaxInliers )
		if( fInlierProb > fMaxInlierProb )
		{

			fMaxInlierProb = fInlierProb;

			iMaxInliers = iInliers;

			feature_to_three_points( pf0 + i*3, pfo0 + i*3*3, pfs0[i], pts0 );
			feature_to_three_points( pf1 + i*3, pfo1 + i*3*3, pfs1[i], pts1 );

			// Calculate transform
			float rot_here0[3][3];
			float rot_here1[3][3];
			float fScaleDiffHere;
			determine_similarity_transform_3point(
						   pts0+0*3, pts0+1*3, pts0+2*3,
						   pts1+0*3, pts1+1*3, pts1+2*3,
						   (float*)(&(rot_here0[0][0])),
						   (float*)(&(rot_here1[0][0])),
						   fScaleDiffHere
						   );

			Feature3DInfo f1Real;
			Feature3DInfo f1Test;

			// Save inliers
			float pfTest[3];
			for( int j = 0; j < iPoints; j++ )
			{
				// Transform point j in image 1
				similarity_transform_3point(
					pf0+j*3, pfTest,
					pf0+i*3, pf1+i*3,
					(float*)(&(rot_here0[0][0])), fScaleDiffHere );

				// Evaluate
				f1Real.x = pf1[j*3+0];
				f1Real.y = pf1[j*3+1];
				f1Real.z = pf1[j*3+2];
				f1Real.scale = pfs1[j];

				f1Test.x = pfTest[0];
				f1Test.y = pfTest[1];
				f1Test.z = pfTest[2];
				f1Test.scale = pfs0[j]*fScaleDiffHere;

				if( piInliers )
				{
					if( compatible_features( f1Real, f1Test, HOUGH_THRES_SCALE, HOUGH_THRES_TRANS ) )
					{
						float rot_temp[3][3];
						memcpy( &(f1Real.ori[0][0]), pfo1 + j*3*3, 3*3*sizeof(float) );
						memcpy( &(rot_temp[0][0]), pfo0 + j*3*3, 3*3*sizeof(float) );
						transpose_3x3<float,float>( rot_temp, f1Test.ori );
						mult_3x3_matrix<float,float>( rot_here0, f1Test.ori, rot_temp );
						transpose_3x3<float,float>( rot_temp, f1Test.ori );
						if( compatible_features( f1Real, f1Test, HOUGH_THRES_SCALE, HOUGH_THRES_TRANS, HOUGH_THRES_ORIEN ) )
						{
							piInliers[j] = 1;
						}
					}
					else
					{
						piInliers[j] = 0;
					}
				}
			}

			// Save output transform parameters

			// Calculate pfC1, by transforming pfC0 -> pfC1
			similarity_transform_3point(
					pfC0, pfC1,
					pf0+i*3, pf1+i*3,
					(float*)(&(rot_here0[0][0])), fScaleDiffHere );

			// Save rotation matrix
			memcpy( rot, &(rot_here0[0][0]), sizeof(rot_here0) );

			// Save scale change
			fScaleDiff = fScaleDiffHere;
		}
	}
	return iMaxInliers;
}

int
MatchKeys(
	vector<Feature3DInfo> &vecImgFeats1,
	vector<Feature3DInfo> &vecImgFeats2,
	vector<int>       &vecModelMatches,
	int  iModelFeatsToConsider,
	char *pcFeatFile1,
	char *pcFeatFile2,
	vector<Feature3DInfo> *pvecImgFeats2Transformed, // Transformed features
	int bMatchConstrainedGeometry, // If true, then consider only approximately correct feature matches

	// Similarity transform parameters
	float *pfScale, // Scale change 2->1, unary
	float *pfRot,   // Rotation 2->1, 3x3 matrix
	float *pfTrans, // Translation 2->1, 1x3 matrix
	int bExpand
)
{
	vecModelMatches.resize(vecImgFeats1.size());

	FEATUREIO fio1;
	FEATUREIO fio2;
	char pcFileName[400];
	char *pch;



	if( iModelFeatsToConsider < 0 )
	{
		iModelFeatsToConsider = vecImgFeats2.size();
	}


	vector<float> pfPoints0;
	vector<float> pfPoints1;
	vector<float> pfScales0;
	vector<float> pfScales1;
	vector<float> pfOris0;
	vector<float> pfOris1;

	vector<int> piInliers;
	int iMatchCount = 0;
	// Save indices
	vector<int> vecIndex0;
	vector<int> vecIndex1;

	// Generate vector of model features

	vector<int> vecMatchIndex12;
	vector<float> vecMatchDist12;

		//if( g_SS.g_nn == -1 )
		//{
		// msInitNearestNeighborDistanceRatioInfoApproximate(vecImgFeats1, 10*vecImgFeats1.size() );
		//}
		//msComputeNearestNeighborDistanceRatioInfoApproximate( vecImgFeats2, vecImgFeats1, vecMatchIndex12, vecMatchDist12 );
		msComputeNearestNeighborDistanceRatioInfo( vecImgFeats2, vecImgFeats1, vecMatchIndex12, vecMatchDist12 );

	vector< pair<int,float> > vecBestMatches;
	for( int i = 0; i < vecMatchIndex12.size(); i++ )
	{
		if( vecMatchIndex12[i] >= 0 )
		{
			vecBestMatches.push_back( pair<int,float>(i,vecMatchDist12[i]));
		}
	}
	sort( vecBestMatches.begin(), vecBestMatches.end(), pairIntFloatLeastToGreatest );

	for( int iImgFeat = 0; iImgFeat < vecImgFeats1.size(); iImgFeat++ )
	{
		Feature3DInfo &featInput = vecImgFeats1[iImgFeat];

		// Flag as unfound...
		vecModelMatches[iImgFeat] = -1;
	}

	int iMaxMatches = 3000;

	vector< float > vecMatchProb;

	//map<float,int> mapModelXtoIndex;
	//map<float,int> mapModelXtoIndexModel;
	//
	//map<float,int> mapInputXtoIndex;
	//map<float,int> mapInputXtoIndexInput;

	map<float,int> mapMatchDuplicateFinder;
	map<float,int> mapMatchDuplicateFinderModel;

	for( int i = 0; i < vecBestMatches.size() && i < iMaxMatches; i++ )
	{
		int iIndex = vecBestMatches[i].first;
		float fDist = vecBestMatches[i].second;
		if( fDist > 0.8 )
		{
			//break;
		}

		Feature3DInfo &featModel = vecImgFeats2[iIndex];
		Feature3DInfo &featInput = vecImgFeats1[ vecMatchIndex12[iIndex] ];

		// Test this hash code - to avoid multiple matches between precisely the same locations
		map<float,int>::iterator itAmazingDupeFinder = mapMatchDuplicateFinder.find(featInput.x*featModel.x);
		if( itAmazingDupeFinder != mapMatchDuplicateFinder.end() )
		{
			//printf( "Found something you both lost!\n" );
			//continue;
		}
		mapMatchDuplicateFinder.insert( pair<float,int>(featInput.x*featModel.x, vecMatchIndex12[iIndex]) );
		mapMatchDuplicateFinderModel.insert( pair<float,int>(featInput.x*featModel.x, iIndex) );


		// Should not allow features with same spatial location

		pfPoints0.push_back(featModel.x);
		pfPoints0.push_back(featModel.y);
		pfPoints0.push_back(featModel.z);



		pfPoints1.push_back(featInput.x);
		pfPoints1.push_back(featInput.y);
		pfPoints1.push_back(featInput.z);

		pfScales0.push_back( featModel.scale );
		pfScales1.push_back( featInput.scale );

		for( int o1=0;o1<3;o1++)
		{
			for( int o2=0;o2<3;o2++ )
			{
				pfOris0.push_back( featModel.ori[o1][o2] );
				pfOris1.push_back( featInput.ori[o1][o2] );
			}
		}

		vecIndex0.push_back( iIndex );
		vecIndex1.push_back( vecMatchIndex12[iIndex] );

		// Save prior probability of this feature
		//    10-02-2015 (October 2, 2015)
		//     It seems we could adjust this threshold to reflect the relative
		//     probability of random feature correspondence, based on feature scale,
		//     the number of features at each scale-aprior and the probability of
		//     of false matches.
		float fFeatsInThreshold = 1;//FeaturesWithinThreshold( vecOrderToFeat[ iIndex ] );
		vecMatchProb.push_back( fFeatsInThreshold );

		iMatchCount++;
	}

	// Now determine transform

	float pfMin[3];
	float pfMax[3];

	getMinMaxDim( vecImgFeats2, pfMin, pfMax );
	float pfModelCenter[3];
	pfModelCenter[0] = (pfMax[0] + pfMin[0])/2.0f;
	pfModelCenter[1] = (pfMax[1] + pfMin[1])/2.0f;
	pfModelCenter[2] = (pfMax[2] + pfMin[2])/2.0f;

	// Model center has no impact on hough transform here - it is only used to parameterize the ouput
	// similarity transform.
	//pfModelCenter[0] = 160.48;
	//pfModelCenter[1] = 147.4;
	//pfModelCenter[2] = 296.58;

	// Output parameters, the result of determine_similarity_transform_ransac()
	float rot[3*3];
	float pfModelCenter2[3];
	float fScaleDiff=1;

	// Save inliers
	for( int iImgFeat = 0; iImgFeat < vecModelMatches.size(); iImgFeat++ )
	{
		vecModelMatches[iImgFeat] = -1;
	}
	int iInliers = -1;

	vector< int > vecModelImgFeatCounts;
	vecModelImgFeatCounts.resize( vecImgFeats2.size(), 0 );

	int iModelImgFeatMaxCount = 0;
	int iModelImgFeatMaxCountIndex = 0;
	if( iMatchCount <= 3 )
	{
		// Not enough matches to determine a solution
		//FILE *outfile = fopen( "result.txt", "a+" );
		//fprintf( outfile, "%d\t%d\n", iMatchCount, 0 );
		//fclose( outfile );
		return iMatchCount;
	}
	piInliers.resize(iMatchCount,0);
	iInliers =
	//determine_similarity_transform_ransac(
	// &(pfPoints0[0]), &(pfPoints1[0]),
	// &(pfScales0[0]), &(pfScales1[0]),
	// &(vecMatchProb[0]),
	// iMatchCount,  500, pfModelCenter,
	// pfModelCenter2, rot, fScaleDiff, &(piInliers[0])
	// );
	determine_similarity_transform_hough(
		&(pfPoints0[0]), &(pfPoints1[0]),
		&(pfScales0[0]), &(pfScales1[0]),
		&(pfOris0[0]), &(pfOris1[0]),
		&(vecMatchProb[0]),
		iMatchCount,  500, pfModelCenter,
		pfModelCenter2, rot, fScaleDiff, &(piInliers[0])
	);


	// Save transform from image 2 to image 1
	if( pfScale )
	*pfScale = fScaleDiff;
	if( pfRot )
	{
		memcpy( pfRot, rot, sizeof(float)*3*3 );
	}
	if( pfTrans )
	{
		float pfStuff[3] = {0,0,0};
		similarity_transform_3point(
			pfStuff, pfTrans,
			pfModelCenter, pfModelCenter2,
			rot, fScaleDiff );
		}

		//FILE *outfile = fopen( "result.txt", "a+" );
		//fprintf( outfile, "%d\t%d\n", iMatchCount, iInliers );
		//fclose( outfile );

		return iInliers;
	}

			int
			removeNonReorientedFeatures(
				vector<Feature3DInfo> &vecFeats
			)
			{
				int iRemoveCount = 0;
				for (int i = 0; i < vecFeats.size(); i++)
				{
					if (!(vecFeats[i].m_uiInfo & INFO_FLAG_REORIENT))
					{
						vecFeats.erase(vecFeats.begin() + i);
						i--;
						iRemoveCount++;
					}
				}
				return iRemoveCount;
			}

			int
			removeReorientedFeatures(
				vector<Feature3DInfo> &vecFeats
			)
			{
				int iRemoveCount = 0;
				for (int i = 0; i < vecFeats.size(); i++)
				{
					if ((vecFeats[i].m_uiInfo & INFO_FLAG_REORIENT))
					{
						vecFeats.erase(vecFeats.begin() + i);
						i--;
						iRemoveCount++;
					}
					else
					{
						memset(&vecFeats[i].ori[0][0], 0, sizeof(vecFeats[i].ori));
						vecFeats[i].ori[0][0] = 1;
						vecFeats[i].ori[1][1] = 1;
						vecFeats[i].ori[2][2] = 1;
					}
				}
				return iRemoveCount;
			}

			int
			removeNonValleyFeatures(
				vector<Feature3DInfo> &vecFeats
			)
			{
				int iRemoveCount = 0;
				for (int i = 0; i < vecFeats.size(); i++)
				{

					if (!(vecFeats[i].m_uiInfo & INFO_FLAG_MIN0MAX1))
					{
						vecFeats.erase(vecFeats.begin() + i);
						i--;
						iRemoveCount++;
					}
				}
				return iRemoveCount;
			}

			int
			removeNonPeakFeatures(
				vector<Feature3DInfo> &vecFeats
			)
			{
				int iRemoveCount = 0;
				for (int i = 0; i < vecFeats.size(); i++)
				{
					if (vecFeats[i].m_uiInfo & INFO_FLAG_MIN0MAX1)
					{
						vecFeats.erase(vecFeats.begin() + i);
						i--;
						iRemoveCount++;
					}
				}
				return iRemoveCount;
			}

			int
			SplitFeatures(
				vector<vector<vector<Feature3DInfo>>> &SplitedFeat,
				int iFeatVec
			)
			{
				int iRemoveCount = 0;
				int peak =0;
				int valley = 0;
				for (int j = 0; j < SplitedFeat[0][iFeatVec].size(); j++)
				{
					if ((SplitedFeat[0][iFeatVec][j].m_uiInfo & INFO_FLAG_MIN0MAX1))
					{
						SplitedFeat[0][iFeatVec].erase(SplitedFeat[0][iFeatVec].begin() + j);
						j--;
						peak++;
					}
				}
				for (int j = 0; j < SplitedFeat[1][iFeatVec].size(); j++)
				{
					if (!(SplitedFeat[1][iFeatVec][j].m_uiInfo & INFO_FLAG_MIN0MAX1))
					{
						SplitedFeat[1][iFeatVec].erase(SplitedFeat[1][iFeatVec].begin() + j);
						j--;
						valley++;
					}
				}
				return iRemoveCount;
			}

			#ifdef INCLUDE_FLANN

			typedef struct _msSearchStructure
			{
				// FLANN parameters
				struct FLANNParameters g_flann_params;
				float g_speedup;
				flann_index_t g_index_id;
				int g_nn;

				// Size of vector
				int iVectorSize;

				// Data vectors
				float *g_pf1; // New image feature
				float *g_pf2; // Data base of features
				int	*g_piFeatureFrequencies; // Frequency count array, one for each feature
				int g_iFeatCount; // Count of all features
				int *g_piFeatureLabels; // Label array, one for each feature, currently set to image index
				int *g_piFeatureImageIndex; // Image index array, one for each feature

				// Label vectors
				vector< float > vfLabelCounts; // A prior over all discrete label values, e.g. in training data

				// Coordinates / Scale for each feature
				float *g_pfX;
				float *g_pfY;
				float *g_pfZ;
				float *g_pfS;

				int	*g_piFeatureL0; // quick label counter
				int	*g_piFeatureL1; // quick label counter

				// Index result
				int *g_piResult; // Result array for one search (input features x g_nn neighours)
				float *g_pfDist;  // Distance array for one search (input features x g_nn neighours)
				int g_iMaxImageFeatures; // Max size of array for one search

				// Indices of features g_pf1
				int *g_piImgIndices; // Index array, one for each image, index mapping image to feature arrays
				int *g_piImgFeatureCounts; // Count array, one for each image, number of features per image
				int g_iImgCount;

				// file output
				FILE *outfile;

				// Labels
			} msSearchStructure;

			msSearchStructure g_SS;

			//
			// msInitNearestNeighborApproximate()
			//
			// Init for NN search for self
			//
			int
			msNearestNeighborApproximateInit(
				vector< vector<Feature3DInfo> > &vvFeats,
				int iNeighbors,
				vector< int > &vLabels,
				float fGeometryWeight,
				int iImgSplit
			)
			{
				g_SS.iVectorSize = PC_ARRAY_SIZE;
				if (fGeometryWeight > 0)
				{
					// Add coordinates to vectors: add
					g_SS.iVectorSize = PC_ARRAY_SIZE + 3;
				}

				// Tells where to split train / test data
				int iFeatureSplit = 0;

				printf("Descriptor size: %d\n", g_SS.iVectorSize);

				g_SS.g_flann_params = DEFAULT_FLANN_PARAMETERS;
				g_SS.g_flann_params.algorithm = FLANN_INDEX_KDTREE;
				g_SS.g_flann_params.trees = 8;
				g_SS.g_flann_params.log_level = FLANN_LOG_INFO;
				g_SS.g_flann_params.checks = 64;
				g_SS.g_flann_params.sorted = 1;
				g_SS.g_flann_params.cores = 1;
				g_SS.g_nn = iNeighbors;

				// Per-image info/arrays: feature indices and counts
				g_SS.g_iImgCount = vvFeats.size();
				g_SS.g_piImgIndices = new int[vvFeats.size()];
				g_SS.g_piImgFeatureCounts = new int[vvFeats.size()];
				g_SS.g_iFeatCount = 0;
				g_SS.g_iMaxImageFeatures = 0;
				for (int i = 0; i < vvFeats.size(); i++)
				{
					if (i == iImgSplit)
					iFeatureSplit = g_SS.g_iFeatCount;

					g_SS.g_piImgIndices[i] = g_SS.g_iFeatCount;
					g_SS.g_piImgFeatureCounts[i] = vvFeats[i].size();
					g_SS.g_iFeatCount += vvFeats[i].size();
					if (vvFeats[i].size() > g_SS.g_iMaxImageFeatures)
					{
						g_SS.g_iMaxImageFeatures = vvFeats[i].size();
					}

					while (vLabels[i] >= g_SS.vfLabelCounts.size())
					{
						// push back zero bins
						g_SS.vfLabelCounts.push_back(0);
					}
					// Add number of features associated with this label
					g_SS.vfLabelCounts[vLabels[i]] += vvFeats[i].size();
				}

				float fPriorSum = 0;
				for (int i = 0; i < g_SS.vfLabelCounts.size(); i++)
				{
					// Add an extra sample to normalize distribution
					g_SS.vfLabelCounts[i]++;
					fPriorSum += g_SS.vfLabelCounts[i];
				}
				// Normalize label distribution - divide by total labels (images) in addition to prior (number of labels)
				for (int i = 0; i < g_SS.vfLabelCounts.size(); i++)
				{
					g_SS.vfLabelCounts[i] /= fPriorSum;
				}

				// Per-feature arrays: labels and descriptors
				g_SS.g_iFeatCount = g_SS.g_iFeatCount;
				g_SS.g_piFeatureLabels = new int[g_SS.g_iFeatCount];
				g_SS.g_piFeatureImageIndex = new int[g_SS.g_iFeatCount];
				g_SS.g_pf2 = new float[g_SS.g_iFeatCount*g_SS.iVectorSize];  // Array to hold entire feature set
				g_SS.g_piFeatureFrequencies = new int[g_SS.g_iFeatCount];  // Array to hold frequency counts for each input feature
				g_SS.g_piFeatureL0 = new int[g_SS.g_iFeatCount]; // Quick label counters
				g_SS.g_piFeatureL1 = new int[g_SS.g_iFeatCount];
				memset(g_SS.g_piFeatureL0, 0, sizeof(int)*g_SS.g_iFeatCount);
				memset(g_SS.g_piFeatureL1, 0, sizeof(int)*g_SS.g_iFeatCount);

				g_SS.g_pfX = new float[g_SS.g_iFeatCount];
				g_SS.g_pfY = new float[g_SS.g_iFeatCount];
				g_SS.g_pfZ = new float[g_SS.g_iFeatCount];
				g_SS.g_pfS = new float[g_SS.g_iFeatCount];

				// Per-result arrays: indices and distances
				g_SS.g_piResult = new int[g_SS.g_iMaxImageFeatures*g_SS.g_nn];
				g_SS.g_pfDist = new float[g_SS.g_iMaxImageFeatures*g_SS.g_nn];
				g_SS.g_pf1 = NULL; // Array to hold max features from one image


				// Copy descriptors into array
				int iFeatCount = 0;
				for (int i = 0; i < vvFeats.size(); i++)
				{
					for (int j = 0; j < vvFeats[i].size(); j++)
					{
						memcpy(g_SS.g_pf2 + iFeatCount*g_SS.iVectorSize, &(vvFeats[i][j].m_pfPC[0]), sizeof(vvFeats[i][j].m_pfPC));

						// This is where the feature info goes - set to zero for debugging
						if (fGeometryWeight > 0)
						{
							g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 0] = fGeometryWeight*vvFeats[i][j].x;
							g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 1] = fGeometryWeight*vvFeats[i][j].y;
							g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 2] = fGeometryWeight*vvFeats[i][j].z;

							g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 0] /= vvFeats[i][j].scale;
							g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 1] /= vvFeats[i][j].scale;
							g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 2] /= vvFeats[i][j].scale;
						}

						//** add geometry weights here: small shift

						g_SS.g_piFeatureLabels[iFeatCount] = vLabels[i];
						g_SS.g_piFeatureImageIndex[iFeatCount] = i;
						g_SS.g_piFeatureFrequencies[iFeatCount] = 0; // Set to zero

						g_SS.g_pfX[iFeatCount] = vvFeats[i][j].x;
						g_SS.g_pfY[iFeatCount] = vvFeats[i][j].y;
						g_SS.g_pfZ[iFeatCount] = vvFeats[i][j].z;
						g_SS.g_pfS[iFeatCount] = vvFeats[i][j].scale;

						iFeatCount++;
					}
				}

				if (iImgSplit <= 0 || iFeatureSplit <= 0)
				iFeatureSplit = g_SS.g_iFeatCount;

				g_SS.g_index_id = flann_build_index(g_SS.g_pf2, iFeatureSplit, g_SS.iVectorSize, &g_SS.g_speedup, &g_SS.g_flann_params);

				g_SS.outfile = fopen("report.all.txt", "wt");
				return 1;
			}

			int
			msNearestNeighborApproximateDelete(
			)
			{
				if (g_SS.outfile) fclose(g_SS.outfile);
				flann_free_index(g_SS.g_index_id, &g_SS.g_flann_params);
				delete[] g_SS.g_piFeatureFrequencies;
				delete[] g_SS.g_pfDist;
				delete[] g_SS.g_piResult;
				delete[] g_SS.g_pf2;

				return 1;
			}


			//
			// msNearestNeighborApproximateSearchSelf()
			//
			// Search a specific image. This function is used in class
			//
			int
			msNearestNeighborApproximateSearchSelf(
				int iImgIndex,
				vector< int > &vfImgCounts,  // Size: #images in database. Return value: # of times features match to database image N
				vector< float > &vfLabelCounts, // Size: #labels in database. Label counts large enough to fit all labels
				vector< float > &vfLabelLogLikelihood, // Size: #labels in database Log likelihood, large enough to fit all labels
				float** ppfMatchingVotes,
				int** ppiLabelVotes
			)
			{
				// Subtract this image label from label prior distribution (add it later)
				// This is important particularly labels with few samples, e.g. nearest neighbors
				int iImgLabel = g_SS.g_piFeatureLabels[g_SS.g_piImgIndices[iImgIndex]];
				float fImgFeaturesProb = g_SS.g_piImgFeatureCounts[iImgIndex] / ((float)(g_SS.g_iFeatCount + g_SS.vfLabelCounts.size()));
				g_SS.vfLabelCounts[iImgLabel] -= fImgFeaturesProb;

				float *pfPriorProbsCT = &(g_SS.vfLabelCounts[0]);

				for (int j = 0; j < vfImgCounts.size(); j++)
				vfImgCounts[j] = 0;
				for (int j = 0; j < vfLabelLogLikelihood.size(); j++)
				vfLabelLogLikelihood[j] = 0;

				int* piResult = new int[g_SS.g_iMaxImageFeatures*g_SS.g_nn];
				float* pfDist = new float[g_SS.g_iMaxImageFeatures*g_SS.g_nn];
				float* pf1 = g_SS.g_pf2 + g_SS.g_piImgIndices[iImgIndex] * g_SS.iVectorSize;

				flann_find_nearest_neighbors_index(g_SS.g_index_id, pf1, g_SS.g_piImgFeatureCounts[iImgIndex], piResult, pfDist, g_SS.g_nn, &g_SS.g_flann_params);

				int iOutOfBounds = 0;

				// No deviation on location filter - if location is relevant (aligned subjects, e.g. brain),
				// then use concatenated geometry
				float fMaxDeviationLocation = 10000000;
				std::map<int, float> votedFeatures;
				for (int i = 0; i < g_SS.g_piImgFeatureCounts[iImgIndex]; i++)
				{
					// Initialize counts with Laplacian prior - actually Laplacian is no good, need actual prior
					for (int j = 0; j < vfLabelCounts.size(); j++)
					vfLabelCounts[j] = 1.0*pfPriorProbsCT[j];

					int iQueryFeatIndex = g_SS.g_piImgIndices[iImgIndex] + i;

					// Identify min distance to neighbor, use this as stdev in Gaussian weighting scheme
					float fMinDist = -1;
					std::vector< std::pair<int, float> > indexNN;
					std::vector< int > matchingImages;
					for (int j = 0; j < g_SS.g_nn; j++)
					{
						int iResultFeatIndex = piResult[i*g_SS.g_nn + j];
						int iLabel = g_SS.g_piFeatureLabels[iResultFeatIndex];
						int iImage = g_SS.g_piFeatureImageIndex[iResultFeatIndex];

						// Geometrical consistency check - not used, fMaxDeviationLocation set to infinity
						if (
							fabs(g_SS.g_pfX[iQueryFeatIndex] - g_SS.g_pfX[iResultFeatIndex]) < fMaxDeviationLocation &&
							fabs(g_SS.g_pfY[iQueryFeatIndex] - g_SS.g_pfY[iResultFeatIndex]) < fMaxDeviationLocation &&
							fabs(g_SS.g_pfZ[iQueryFeatIndex] - g_SS.g_pfZ[iResultFeatIndex]) < fMaxDeviationLocation &&
							1
						)
						{
							// Result index must not be from the query image
							if (iResultFeatIndex < g_SS.g_piImgIndices[iImgIndex] || iResultFeatIndex > g_SS.g_piImgIndices[iImgIndex] + g_SS.g_piImgFeatureCounts[iImgIndex])
							{
								// Only count features from other images
								if (indexNN.size() < g_SS.g_nn)
								{
									// Ensure feature only vote once per image
									if (std::find(matchingImages.begin(), matchingImages.end(), iImage) == matchingImages.end())
									{
										/* A vote does not already exists for this image */
										std::pair<int, float> tmp(piResult[i*g_SS.g_nn + j], pfDist[i*g_SS.g_nn + j]);
										indexNN.push_back(tmp);

										if (fMinDist == -1 || pfDist[i*g_SS.g_nn + j] < fMinDist)
										{
											if (pfDist[i*g_SS.g_nn + j] > 0)
											{
												// For duplicated scans, distance will be 0, we want first non-null minimum distance
												fMinDist = pfDist[i*g_SS.g_nn + j];
											}
										}

										matchingImages.push_back(iImage);
									}
								}
								else {
									break;
								}
							}
						}
						else
						{
							// Should not get here with huge permissible deviation
							assert(0);
						}
					}


					// Normalize weights
					float fSumWeights = 0;
					std::vector<float> weights;
					for (int j = 0; j < indexNN.size(); ++j)
					{
						int iResultFeatIndex = indexNN[j].first;

						float fDx = (g_SS.g_pfX[iQueryFeatIndex] - g_SS.g_pfX[iResultFeatIndex]);
						float fDy = (g_SS.g_pfY[iQueryFeatIndex] - g_SS.g_pfY[iResultFeatIndex]);
						float fDz = (g_SS.g_pfZ[iQueryFeatIndex] - g_SS.g_pfZ[iResultFeatIndex]);
						float fSc1 = g_SS.g_pfS[iQueryFeatIndex];
						float fSc2 = g_SS.g_pfS[iResultFeatIndex];

						// ----------------------------------------------------------------------------------------
						// Appearance Weight

						float fDistApp = indexNN[j].second;
						float fDistSqApp = fDistApp*fDistApp;

						float fVarApp = fMinDist*fMinDist;

						float fAppWeight = std::exp(-fDistSqApp / fVarApp);

						// ----------------------------------------------------------------------------------------
						// Total Weight

						float fTotalWeight = fAppWeight;

						weights.push_back(fTotalWeight);
						fSumWeights += fTotalWeight;
					}

					// Avoid division-by-0
					if (fSumWeights <= 0) {
						continue;
					}

					// SoftMax + log
					float eta = 1; // Background distribution
					for (int j = 0; j < weights.size(); ++j)
					{
						weights[j] /= fSumWeights;

						weights[j] += eta;
						weights[j] = std::log(weights[j]);
						weights[j] /= std::log(eta + 1);
					}

					// Now add results based on min distance neighbor
					for (int j = 0; j < indexNN.size(); ++j)
					{
						int iResultFeatIndex = indexNN[j].first;
						int iLabel = g_SS.g_piFeatureLabels[iResultFeatIndex];
						int iImage = g_SS.g_piFeatureImageIndex[iQueryFeatIndex];

						float fDx = (g_SS.g_pfX[iQueryFeatIndex] - g_SS.g_pfX[iResultFeatIndex]);
						float fDy = (g_SS.g_pfY[iQueryFeatIndex] - g_SS.g_pfY[iResultFeatIndex]);
						float fDz = (g_SS.g_pfZ[iQueryFeatIndex] - g_SS.g_pfZ[iResultFeatIndex]);
						float fDistGeom = fDx*fDx + fDy*fDy + fDz*fDz;
						if (fDistGeom > 0)
						fDistGeom = sqrt(fDistGeom);


						// Geometrical consistency check
						if (
							fabs(g_SS.g_pfX[iQueryFeatIndex] - g_SS.g_pfX[iResultFeatIndex]) < fMaxDeviationLocation &&
							fabs(g_SS.g_pfY[iQueryFeatIndex] - g_SS.g_pfY[iResultFeatIndex]) < fMaxDeviationLocation &&
							fabs(g_SS.g_pfZ[iQueryFeatIndex] - g_SS.g_pfZ[iResultFeatIndex]) < fMaxDeviationLocation &&
							1
						)
						{
							if (iResultFeatIndex < g_SS.g_piImgIndices[iImgIndex] || iResultFeatIndex > g_SS.g_piImgIndices[iImgIndex] + g_SS.g_piImgFeatureCounts[iImgIndex])
							{
								// Only count features from other images
								float fDist = pfDist[i*g_SS.g_nn + j];
								float fExponent = fDist / (fMinDist + 1.0);
								// Gaussian weighting
								float fValue = exp(-fExponent*fExponent) / pfPriorProbsCT[iLabel];
								vfLabelCounts[iLabel] += fValue;

								// Multiple features from same subject cannot match to same feature in another subject (e.g. A1 -> B1, A2 -> B1)
								if (votedFeatures.find(iResultFeatIndex) != votedFeatures.end())
								{
									float previousVote = votedFeatures[iResultFeatIndex];

									// If this vote is better, delete previous one
									if (weights[j] > previousVote)
									{

										if (previousVote > 0) {
											ppfMatchingVotes[iImage][iLabel] -= previousVote;
										}
										ppfMatchingVotes[iImage][iLabel] += weights[j];
										votedFeatures[iResultFeatIndex] = weights[j];
									}
								}
								else
								{
									// Store votes
									ppfMatchingVotes[iImage][iLabel] += weights[j];
									ppiLabelVotes[iImage][iLabel] += 1;
									votedFeatures.insert({ iResultFeatIndex, weights[j] });
								}

								vfImgCounts[iLabel]++;
							}
						}
						else
						{
							// Should not get here with huge permissible deviation
							assert(0);
						}
					}

					// Compute & accumulate log likelihood
					float fTotal = 0;
					for (int j = 0; j < vfLabelCounts.size(); j++)
					{
						// Divide here by prior probability of a sample
						fTotal += vfLabelCounts[j];
					}
					// Note that normalization here makes no difference immediately
					for (int j = 0; j < vfLabelCounts.size(); j++)
					{
						vfLabelLogLikelihood[j] += log(vfLabelCounts[j] / (float)fTotal);
					}
				}

				// Add label back on
				g_SS.vfLabelCounts[iImgLabel] += fImgFeaturesProb;

				delete[] piResult;
				delete[] pfDist;

				return 1;
			}

			#endif
