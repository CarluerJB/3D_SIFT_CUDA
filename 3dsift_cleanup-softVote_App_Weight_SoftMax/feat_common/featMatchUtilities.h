

#ifndef __FEATMATCHUTILITIES_H__
#define __FEATMATCHUTILITIES_H__

#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <map>

#include "MultiScale.h"

using namespace std;

#define LOG_1_5 0.4054651

int
determine_rotation_3point(
					   float *pfP01, float *pfP02, float *pfP03, // points in image 1 (3d)
					   float *rot // rotation matrix  (3x3)
					   );

int
determine_rotation_3point(
					   float *pfP01, float *pfP02, float *pfP03, // points in image 1 (3d)
					   float *pfP11, float *pfP12, float *pfP13, // points in image 2 (3d)
					   float *rot // rotation matrix  (3x3)
					   );

int
compatible_features(
					const Feature3DInfo &f1,
					const Feature3DInfo &f2,
					//float fScaleDiffThreshold = (float)log(1.5f), // works the best, MICCAI 2009 results
					float fScaleDiffThreshold = LOG_1_5,
					float fShiftThreshold = 0.5, // works the best, MICCAI 2009 results
					float fCosineAngleThreshold = -1.0f // Allow all all angles by default
					);

void
vec3D_euler_rotation(
				   float *pfx0, // 1x3
				   float *pfy0, // 1x3
				   float *pfz0, // 1x3

				   float *pfx1, // 1x3
				   float *pfy1, // 1x3
				   float *pfz1, // 1x3

				   float *ori // 3x3
				   );

int
removeZValue(
							vector<Feature3DInfo> &vecFeats
							);

int
removeNonReorientedFeatures(
							vector<Feature3DInfo> &vecFeats
							);

int
removeNonPeakFeatures(
	vector<Feature3DInfo> &vecFeats
);

int
removeNonValleyFeatures(
	vector<Feature3DInfo> &vecFeats
);

int
SplitFeatures(
	vector<vector< vector<Feature3DInfo> >> &SplitedFeat,
	int iFeatVec
);

int
removeReorientedFeatures(
							vector<Feature3DInfo> &vecFeats
							);

int
MatchKeys(
		vector<Feature3DInfo> &vecImgFeats1,
		vector<Feature3DInfo> &vecImgFeats2,
		vector<int>       &vecModelMatches,
		int				  iModelFeatsToConsider = -1,
		char *pcFeatFile1 = 0,
		char *pcFeatFile2 = 0,
		vector<Feature3DInfo> *pvecImgFeats2Transformed = 0, // Transformed features
		int		bMatchConstrainedGeometry = 0, // If true, then consider only approximately correct feature matches

		// Similarity transform parameters
		float *pfScale=0, // Scale change 2->1, unary
		float *pfRot=0,   // Rotation 2->1, 3x3 matrix
		float *pfTrans=0, // Translation 2->1, 1x3 matrix
		int bExpand=0	// Expanded Hough transform - finds more correspondences
		);

int
msComputeNearestNeighborDistanceRatioInfo(
					 vector<Feature3DInfo> &vecFeats1,
					 vector<Feature3DInfo> &vecFeats2,
					 vector<int>		&vecMatchIndex12,
					 vector<float>		&vecMatchDist12
						  );

int
msComputeNearestNeighborDistanceRatioInfoApproximate(
					 vector<Feature3DInfo> &vecFeats1,
					 vector<Feature3DInfo> &vecFeats2,
					 vector<int>		&vecMatchIndex12,
					 vector<float>		&vecMatchDist12
						  );

//
// msComputeNearestNeighborDistanceRatioInfoApproximate()
//
// For fast computation.
//
int
msComputeNearestNeighborDistanceRatioInfoApproximate(
				float *pf1, float *pf2,
				int iSize1, int iSize2,
				int *piResult,
				float *pfDist
				);

//
// DetermineTransform()
//
// Determine transform from matches.
//
int
DetermineTransform(
		vector<Feature3DInfo> &vecImgFeats1,
		vector<Feature3DInfo> &vecImgFeats2,
		vector<int>       &vecModelMatches,

		// Similarity transform parameters
		float *pfScale, // Scale change 2->1, unary
		float *pfRot,   // Rotation 2->1, 3x3 matrix
		float *pfTrans, // Translation 2->1, 1x3 matrix
		char *pcFeatFile1,
		char *pcFeatFile2
		);

class TransformSimilarity
{
public:
	TransformSimilarity()
	{
		Identity();
	};

	~TransformSimilarity() {};

	//
	// Identity()
	//
	// Create identity matrix.
	//
	void
	Identity()
	{
		m_fScale = 1;
		for( int i = 0; i < 3; i++ )
		{
			for( int j = 0; j < 3; j++ )
			{
				m_ppfRot[i][j] = 0;
			}
			m_ppfRot[i][i] = 1;
			m_pfTrans[i] = 0;
		}
	}

	//
	// Multiply()
	//
	// Multiply this similarity transform from the left.
	//
	void
	Multiply(
		TransformSimilarity &tsLeft
		)
	{
		float rot_here[3][3];
		memcpy( &(rot_here[0][0]), &(m_ppfRot[0][0]), sizeof(m_ppfRot) );

		mult_3x3_matrix<float,float>( tsLeft.m_ppfRot, rot_here, m_ppfRot );

		m_fScale *= tsLeft.m_fScale;

		float pfTransHere[3];
		memcpy( pfTransHere, m_pfTrans, sizeof(m_pfTrans) );
		for( int r = 0; r < 3; r++ )
		{
			m_pfTrans[r] = 0;
			for( int c = 0; c < 3; c++ )
			{
				m_pfTrans[r] += tsLeft.m_fScale*tsLeft.m_ppfRot[r][c]*pfTransHere[c];
			}
			m_pfTrans[r] += tsLeft.m_pfTrans[r];
		}
	}

	void
	Invert(
	)
	{
		float zero_vec[3] = {0,0,0};
		float stuff[3] = {0,0,0};
		similarity_transform_invert( stuff, m_pfTrans, m_ppfRot[0], m_fScale );

		float new_trans[3] = {0,0,0};
		similarity_transform_3point(
					zero_vec, new_trans,
					stuff, m_pfTrans,
					m_ppfRot[0], m_fScale );
		memcpy( m_pfTrans, new_trans, sizeof(new_trans) );
	}

	int
	ReadMatrix(
		char *pcFileName
	)
	{
		int iReturn;
		FILE *infile = fopen( pcFileName, "rt" );
		for( int r = 0; r < 3; r++ )
		{
			float fSumSqr = 0;
			for( int c = 0; c < 3; c++ )
			{
				iReturn = fscanf( infile, "%f\t", &m_ppfRot[r][c] );
				if( iReturn != 1 )
					return -1;
			}
			iReturn = fscanf( infile, "%f\n", &m_pfTrans[r] );
			if( iReturn != 1 ) return -1;
		}
		fclose( infile );

		float fAvgMag = 0;
		for( int c = 0; c < 3; c++ )
		{
			float fSumSqr = 0;
			for( int r = 0; r < 3; r++ )
			{
				fSumSqr += m_ppfRot[r][c]*m_ppfRot[r][c];
			}
			if( fSumSqr <= 0 )
			{
				return -1;
			}
			fSumSqr = sqrt( fSumSqr );
			for( int r = 0; r < 3; r++ )
			{
				m_ppfRot[r][c] /= fSumSqr;
			}
			fAvgMag += fSumSqr;
		}

		m_fScale = fAvgMag/3;

		return 1;
	}

	void
	WriteMatrix(
		char *pcFileName
	)
	{
		FILE *outfile = fopen( pcFileName, "wt" );
		for( int r = 0; r < 3; r++ )
		{
			for( int c = 0; c < 3; c++ )
			{
				fprintf( outfile, "%f\t", m_fScale*m_ppfRot[r][c] );
			}
			fprintf( outfile, "%f\n", m_pfTrans[r] );
		}
		fprintf( outfile, "0.0\t0.0\t0.0\t1.0\n" );
		fclose( outfile );
	}

	float m_fScale;
	float m_ppfRot[3][3];
	float m_pfTrans[3];
};


int
MatchKeysDeformationConstrained(
		vector<Feature3DInfo> &vecImgFeats1,
		vector<Feature3DInfo> &vecImgFeats2,
		vector<int>       &vecModelMatches,
		int				  iModelFeatsToConsider,
		char *pcFeatFile1,
		char *pcFeatFile2,
		FEATUREIO &fioDef
		);



//
// October 2014 - 2ndNeighbor functions introduced to identify similar content in images, for duplicate feature detection.
//
// MatchKeysNearest2ndNeighbor()
//    Return a list of 2nd nearest neighbors.
//
int
MatchKeysNearest2ndNeighbor(
		vector<Feature3DInfo> &vecImgFeats1,
		vector<Feature3DInfo> &vecImgFeats2,
		char *pcFeatFile1,
		char *pcFeatFile2
		);



//
// Approximate nearest neighbor routines.
//
//
//
//

int
msNearestNeighborApproximateInit(
					 vector< vector<Feature3DInfo> > &vvFeats,
					 int iNeighbors,
					 vector< int > &vLabels,
					 float fGeometryWeight = -1, // Optional weight for feature geometry, if negative only appearance is used
					 int iImgSplit = -1 // Test learning to leave out all images split
	);

int
msNearestNeighborApproximateDelete(
	);

int
msNearestNeighborApproximateSearchSelf(
						  );

int
msNearestNeighborApproximateSearchSelf(
	int iImgIndex,
	vector< int > &viImgCounts,  // Size: #images in database. Return value: # of times features match to database image N
	vector< float > &vfLabelCounts, // Label counts large enough to fit all labels
	vector< float > &vfLabelLogLikelihood, // Log likelihood, large enough to fit all labels
	float** ppfMatchingVotes,
	int** ppiLabelVotes
);

int
msNearestNeighborApproximateBarfInards(
);

#endif
