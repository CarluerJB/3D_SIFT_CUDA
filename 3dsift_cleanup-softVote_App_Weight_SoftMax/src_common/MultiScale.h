#ifndef __MULTISCALE_H__
#define __MULTISCALE_H__

#pragma warning(disable:4786)

#include <vector>
#include <cstring>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "FeatureIO.h"
#include "LocationValue.h"

using namespace std;

// Version 001: 48 principal components added to info file
#define FEAT3DHEADER "FEAT3D001"
#define FEAT2DHEADER "FEAT2D001"

//
// Feature3D
// A 3D scale-invariant feature.
//

class Feature3D;

// Defines wether feature corresponds to a minima (0) or maxima (1)
#define INFO_FLAG_MIN0MAX1 0x00000010
// Defines wether feature appearance has been reoriented, yes (1) or no (0)
#define INFO_FLAG_REORIENT 0x00000020
// Defines wether feature is a line feature (1) or not (0)
// In the case of a line feature, the ori field contains the coordinates
// of the 2nd point
#define INFO_FLAG_LINE     0x00000100

// 12:34 PM 4/27/2011
// Make array big enough to store gradient orientation histograms spatial locations=(2x2x2), orientations=(8)
#define PC_ARRAY_SIZE 64

typedef float PC_ARRAY[PC_ARRAY_SIZE];

class Feature3DInfo
{

public:

	Feature3DInfo();

	//
	// Visualization functions
	//
	int
		OutputVisualFeature(
			char *pcFileNameBase,
			int bOneImage = 1 // Output 3 planes in 1 image
		) {
		return 1;
	}

	float
	DistSqrPCs(
			const Feature3DInfo &feat3D,
			int iPCs
		) const
	{
		float fSumSqr = 0;
		for( int i = 0; i < iPCs; i++ )
		{
			float fDiff = (m_pfPC[i] - feat3D.m_pfPC[i]);
			fSumSqr += fDiff*fDiff;
		}
		return fSumSqr;
	}

	// Rank order only PC components
	void
		NormalizeDataRankedPCs(
		);

	void
		ZeroData(
		);

	// Apply similarity transform in matrix form
	void
		SimilarityTransform(
			float *pfMat4x4
		);

	//
	//
	//
	//
	//
	int
		DeepCopy(
			Feature3DInfo &feat3D
		)
	{
		m_uiInfo = feat3D.m_uiInfo;
		x = feat3D.x;
		y = feat3D.y;
		z = feat3D.z;
		scale = feat3D.scale;
		std::memcpy(&(m_pfPC[0]), &(feat3D.m_pfPC[0]), sizeof(m_pfPC));
		std::memcpy(&(ori[0][0]), &(feat3D.ori[0][0]), sizeof(ori));
		std::memcpy(&(eigs[0]), &(feat3D.eigs[0]), sizeof(eigs));
		return 0;
	}

	// Information regarding this feature
	unsigned int	m_uiInfo;

	// Location
	float x;
	float y;
	float z;
	// Scale
	float scale;
	// Orientation axes (major to minor)
	float ori[3][3];
	// Eigenvalues (major to minor)
	float eigs[3];

	// Insert principal components into info .. it is very small compared to
	// size of data_zyx ...
	static const int FEATURE_3D_PCS = PC_ARRAY_SIZE;
	// Data fields: added to support different data descriptors
	float m_pfPC[FEATURE_3D_PCS];
};

class Feature3DData
{

public:
	// Image sample dimension
	static const int FEATURE_3D_DIM = 11;

	float data_zyx[FEATURE_3D_DIM][FEATURE_3D_DIM][FEATURE_3D_DIM];
};

class Feature3D : public Feature3DInfo, public Feature3DData
{
public:

	Feature3D() {};
	~Feature3D() {};

	//
	// NormalizeData()
	//
	// Normalize data vector to unit Euclidean length.
	// Subtract mean.
	//
	// Optionally, return mean & varr of original data.
	//
	void
		NormalizeData(
			float *pfMean = 0,
			float *pfVarr = 0,
			float *pfMin = 0,
			float *pfMax = 0
		);
};

typedef struct _EXTREMA_STRUCT
{
	FEATUREIO *pfioH;
	FEATUREIO *pfioL;
	LOCATION_VALUE_XYZ lva;
	int iCurrIndex;
	int *piNeighbourIndices;
	int iNeighbourCount;
	int bDense;
} EXTREMA_STRUCT;

typedef int EXTREMA_FUNCTION(EXTREMA_STRUCT *);

typedef struct _COMP_MATCH_VALUE
{
	int iIndex1;
	int iIndex2;
	float fValue;
} COMP_MATCH_VALUE;


//
// inverse_3x3()
//
// Invert a 3x3 matrix.
//
template <class T, class DIV>
void
invert_3x3(
	T mat_in[3][3],
	T mat_out[3][3]
)
{
	T &a11 = mat_in[0][0];
	T &a21 = mat_in[1][0];
	T &a31 = mat_in[2][0];
	T &a12 = mat_in[0][1];
	T &a22 = mat_in[1][1];
	T &a32 = mat_in[2][1];
	T &a13 = mat_in[0][2];
	T &a23 = mat_in[1][2];
	T &a33 = mat_in[2][2];

	T det = a11*(a33*a22 - a32*a23) - a21*(a33*a12 - a32*a13) + a31*(a23*a12 - a22*a13);

	DIV div = 1 / (DIV)det;

	mat_out[0][0] = (a33*a22 - a32*a23)*div;
	mat_out[1][0] = -(a33*a21 - a31*a23)*div;
	mat_out[2][0] = (a32*a21 - a31*a22)*div;
	mat_out[0][1] = -(a33*a12 - a32*a13)*div;
	mat_out[1][1] = (a33*a11 - a31*a13)*div;
	mat_out[2][1] = -(a32*a11 - a31*a12)*div;
	mat_out[0][2] = (a23*a12 - a22*a13)*div;
	mat_out[1][2] = -(a23*a11 - a21*a13)*div;
	mat_out[2][2] = (a22*a11 - a21*a12)*div;
}

template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

template< class FEATURE_TYPE >
int
msFeature3DVectorOutputBin(
	vector<FEATURE_TYPE> &vecFeats3D,
	char *pcFileName,
	float fEigThres = -1
)
{
	FILE *outfile = fopen(pcFileName, "wb");
	if (!outfile)
	{
		return -1;
	}
	int iFeatCount = 0;

	for (int i = 0; i < vecFeats3D.size(); i++)
	{
		FEATURE_TYPE &feat3D = vecFeats3D[i];

		// Sphere, apply threshold
		float fEigSum = feat3D.eigs[0] + feat3D.eigs[1] + feat3D.eigs[2];
		float fEigPrd = feat3D.eigs[0] * feat3D.eigs[1] * feat3D.eigs[2];
		float fEigSumProd = fEigSum*fEigSum*fEigSum;
		if (fEigSumProd < fEigThres*fEigPrd || fEigThres < 0)
		{
			iFeatCount++;
		}
	}

	fprintf(outfile, "# featExtract %s\n", "1.1");
	fprintf(outfile, "Features: %d\n", iFeatCount);

	for (int i = 0; i < vecFeats3D.size(); i++)
	{
		FEATURE_TYPE &feat3D = vecFeats3D[i];

		// Sphere, apply threshold
		float fEigSum = feat3D.eigs[0] + feat3D.eigs[1] + feat3D.eigs[2];
		float fEigPrd = feat3D.eigs[0] * feat3D.eigs[1] * feat3D.eigs[2];
		float fEigSumProd = fEigSum*fEigSum*fEigSum;
		if (fEigSumProd < fEigThres*fEigPrd || fEigThres < 0)
		{
		}
		else
		{
			continue;
		}

		// Location and scale
		fwrite(&(vecFeats3D[i].x), sizeof(float), 1, outfile);
		fwrite(&(vecFeats3D[i].y), sizeof(float), 1, outfile);
		fwrite(&(vecFeats3D[i].z), sizeof(float), 1, outfile);
		fwrite(&(vecFeats3D[i].scale), sizeof(float), 1, outfile);

		// Orientation (could save a vector here)
		fwrite(&(vecFeats3D[i].ori[0][0]), sizeof(float), 9, outfile);

		// Eigenvalues of 2nd moment matrix
		fwrite(&(vecFeats3D[i].eigs[0]), sizeof(float), 3, outfile);

		// Info flag
		fwrite(&(vecFeats3D[i].m_uiInfo), sizeof(unsigned int), 1, outfile);

		// Output principal components, if set
		unsigned char pucVec[Feature3DInfo::FEATURE_3D_PCS];
		for (int j = 0; j < Feature3DInfo::FEATURE_3D_PCS; j++)
		{

			pucVec[j] = (unsigned char)(vecFeats3D[i].m_pfPC[j]);
		}
		fwrite(&(pucVec[0]), sizeof(unsigned char), Feature3DInfo::FEATURE_3D_PCS, outfile);
	}

	fclose(outfile);
	return 0;
}

template< class FEATURE_TYPE >
int
msFeature3DVectorInputText(
	vector<FEATURE_TYPE> &vecFeats3D,
	char *pcFileName,
	float fEigThres = -1
)
{
	FILE *infile = fopen(pcFileName, "rt");
	if (!infile)
	{
		return -1;
	}
	int iFeatCount = vecFeats3D.size();

	char buff[400];
	buff[0] = '#';

	// Read past comments
	while (buff[0] == '#')
	{
		fgets(buff, sizeof(buff), infile);
	}

	if (sscanf(buff, "Features: %d\n", &iFeatCount) <= 0 || iFeatCount <= 0)
	{
		fclose(infile);
		return -1;
	}

	fgets(buff, sizeof(buff), infile);
	if (!strstr(buff, "Scale-space location[x y z scale]"))
	{
		fclose(infile);
		return -1;
	}

	vecFeats3D.resize(iFeatCount);

	for (int i = 0; i < iFeatCount; i++)
	{
		int iReturn =
			fscanf(infile, "%f\t%f\t%f\t%f\t", &(vecFeats3D[i].x), &(vecFeats3D[i].y), &(vecFeats3D[i].z), &(vecFeats3D[i].scale));
		assert(iReturn == 4);

		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				iReturn = fscanf(infile, "%f\t", &(vecFeats3D[i].ori[j][k]));
				assert(iReturn == 1);
			}
		}

		// Eigenvalues of 2nd moment matrix
		for (int j = 0; j < 3; j++)
		{
			iReturn = fscanf(infile, "%f\t", &(vecFeats3D[i].eigs[j]));
			assert(iReturn == 1);
		}

		// Info flag
		iReturn = fscanf(infile, "%d\t", &(vecFeats3D[i].m_uiInfo));
		assert(iReturn == 1);

		for (int j = 0; j < Feature3DInfo::FEATURE_3D_PCS; j++)
		{
			iReturn = fscanf(infile, "%f\t", &(vecFeats3D[i].m_pfPC[j]));
			assert(iReturn == 1);
		}

		FEATURE_TYPE &feat3D = vecFeats3D[i];
	}

	fclose(infile);

	vecFeats3D.resize(iFeatCount);

	return 0;
}

template< class FEATURE_TYPE >
int
msFeature3DVectorOutputText(
	vector<FEATURE_TYPE> &vecFeats3D,
	char *pcFileName,
	float fEigThres = -1,
	int iCommentLines = 0,
	char **ppcCommentLines = 0
)
{
	FILE *outfile = fopen(pcFileName, "wt");
	if (!outfile)
	{
		return -1;
	}
	int iFeatCount = 0;

	for (int i = 0; i < vecFeats3D.size(); i++)
	{
		FEATURE_TYPE &feat3D = vecFeats3D[i];

		// Sphere, apply threshold
		float fEigSum = feat3D.eigs[0] + feat3D.eigs[1] + feat3D.eigs[2];
		float fEigPrd = feat3D.eigs[0] * feat3D.eigs[1] * feat3D.eigs[2];
		float fEigSumProd = fEigSum*fEigSum*fEigSum;
		if (fEigSumProd < fEigThres*fEigPrd || fEigThres < 0)
		{
			iFeatCount++;
		}
	}

	fprintf(outfile, "# featExtract %s\n", "1.1");
	for (int i = 0; i < iCommentLines; i++)
	{
		// Print out comments
		fprintf(outfile, "# %s\n", ppcCommentLines[i]);
	}
	fprintf(outfile, "Features: %d\n", iFeatCount);
	fprintf(outfile, "Scale-space location[x y z scale] orientation[o11 o12 o13 o21 o22 o23 o31 o32 o32] 2nd moment eigenvalues[e1 e2 e3] info flag[i1] descriptor[d1 .. d64]\n");

	for (int i = 0; i < vecFeats3D.size(); i++)
	{
		FEATURE_TYPE &feat3D = vecFeats3D[i];

		// Sphere, apply threshold
		float fEigSum = feat3D.eigs[0] + feat3D.eigs[1] + feat3D.eigs[2];
		float fEigPrd = feat3D.eigs[0] * feat3D.eigs[1] * feat3D.eigs[2];
		float fEigSumProd = fEigSum*fEigSum*fEigSum;
		if (fEigSumProd < fEigThres*fEigPrd || fEigThres < 0)
		{
		}
		else
		{
			continue;
		}

		// Location and scale
		fprintf(outfile, "%f\t%f\t%f\t%f\t", vecFeats3D[i].x, vecFeats3D[i].y, vecFeats3D[i].z, vecFeats3D[i].scale);

		// Orientation (could save a vector here)
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				fprintf(outfile, "%f\t", vecFeats3D[i].ori[j][k]);
			}
		}

		// Eigenvalues of 2nd moment matrix
		for (int j = 0; j < 3; j++)
		{
			fprintf(outfile, "%f\t", vecFeats3D[i].eigs[j]);
		}

		// Info flag
		fprintf(outfile, "%d\t", vecFeats3D[i].m_uiInfo);

		// Output principal components, if set
		for (int j = 0; j < Feature3DInfo::FEATURE_3D_PCS; j++)
		{
			//fprintf(outfile, "%f\t", (vecFeats3D[i].m_pfPC[j]));
			fprintf(outfile, "%i\t", (char)(vecFeats3D[i].m_pfPC[j]));
		}
		fprintf(outfile, "\n");
	}

	fclose(outfile);
	return 0;
}

void
vec3D_norm_3d(
	float *pf1
);


float
vec3D_mag(
	float *pf1
);

void
mult_4x4_vector(
	float *mat,
	float *vec_in,
	float *vec_out
);

template <class T, class DIV>
void
mult_3x3(
	T mat[3][3],
	T vec_in[3],
	T vec_out[3]
)
{
	for (int i = 0; i < 3; i++)
	{
		vec_out[i] = 0;
		for (int j = 0; j < 3; j++)
		{
			vec_out[i] += mat[i][j] * vec_in[j];
		}
	}
}

template <class T, class DIV>
void
mult_3x3_matrix(
	T mat1[3][3],
	T mat2[3][3],
	T mat_out[3][3]
)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			mat_out[i][j] = 0;
			for (int ii = 0; ii < 3; ii++)
			{
				mat_out[i][j] += mat1[i][ii] * mat2[ii][j];
			}
		}
	}
}
void display(int start, int nb);

int
msGeneratePyramidDOG3D_efficient(
	FEATUREIO	&fioImg,
	vector<Feature3D> &vecFeats,
	int best_device_id,
	float fInitialImageScale = 1.0f,
	int bDense = 0,
	int bOutput = 1,
	float fEigThres = -1 // Eigenvalue threshold, ignore if < 0
);

//
// Resample feature appearance as gradient orientation histograms. data_xyz must
// contain sampled data.
//
int
msResampleFeaturesGradientOrientationHistogram(
	Feature3D &feat3D
);

//
// msResampleFeaturesBRIEF()
//
// Resample feature appearance as BRIEF Descriptor
// (Binary Robust Independent Elementary Features). data_xyz must contain
// sampled data.
//
int
msResampleFeaturesBRIEF(
	Feature3D &feat3D,
	vector<LOCATION_VALUE_XYZ> &RandomIndexX,
	vector<LOCATION_VALUE_XYZ> &RandomIndexY
);

int hammingDistance(int n1, int n2);

int sphericalToCartesian(float fmagn, float fZenithAng, float fAzimuthAng, int &x, int &y, int &z);

int
msGenerateBRIEFindex(
	vector<LOCATION_VALUE_XYZ> &RandomIndexX,
	vector<LOCATION_VALUE_XYZ> &RandomIndexY,
	int iCount,
	FEATUREIO fio,
	int method=2
);

int
euclidean_distance_3d(
	int x1,
	int x2,
	int y1,
	int y2,
	int z1,
	int z2);

static int
validateDifferencePeak3D(
	LOCATION_VALUE_XYZ &lvPeak,
	FEATUREIO &fio,
	FEATUREIO &fio2
);

static int
validateDifferenceValley3D(
	LOCATION_VALUE_XYZ &lvPeak,
	FEATUREIO &fio,
	FEATUREIO &fio2
);

int
generateFeatures3D_efficient(
	LOCATION_VALUE_XYZ_ARRAY	&lvaMinima,
	LOCATION_VALUE_XYZ_ARRAY	&lvaMaxima,

	vector<float> &vecMinH,
	vector<float> &vecMinL,
	vector<float> &vecMaxH,
	vector<float> &vecMaxL,

	FEATUREIO &fioC,	// Center image: for interpolating feature geometry

	float fScaleH,	// Scale factors: for interpolating feature geometry
	float fScaleC,
	float fScaleL,

	FEATUREIO &fioImg,

	float fScale,	// Equals fScaleC
	vector<Feature3D> &vecFeats,
	float fEigThres = -1
);

//
// generateFeatures3D()
//
// Doesn't consider image data, strictly outputs max
// This version is designed for multi-modal feature selection and registration,
// here we output prior histograms to be used in registration.
//
int
generateFeatures3D(
	LOCATION_VALUE_XYZ_ARRAY	&lvaMinima,
	LOCATION_VALUE_XYZ_ARRAY	&lvaMaxima,

	FEATUREIO &fioH,	// Hi res image
	FEATUREIO &fioC,	// Center image: for interpolating feature geometry
	FEATUREIO &fioL,	// Lo res image

	float fScaleH,	// Scale factors: for interpolating feature geometry
	float fScaleC,
	float fScaleL,
	float fScale,

	FEATUREIO *fioImages,
	int iImages,

	vector<Feature3D> &vecFeats
);

//
// detectExtrema4D_test_interleave()
//
// Determine if Extrema detection can be launch on gpu
int
detectExtrema4D_test_interleave(
	FEATUREIO fioD0, FEATUREIO fioD1, FEATUREIO fioSumOfSign,
	LOCATION_VALUE_XYZ_ARRAY &lvaMinima,
	LOCATION_VALUE_XYZ_ARRAY &lvaMaxima,
	int best_device_id
);

int
detectExtrema4D_test(
	FEATUREIO *pfioH,	// Hi res image
	FEATUREIO *pfioC,	// Center image
	FEATUREIO *pfioL,	// Lo res image
	LOCATION_VALUE_XYZ_ARRAY	&lvaMinima,
	LOCATION_VALUE_XYZ_ARRAY	&lvaMaxima,
	int bDense = 0
);

float
vec3D_dot_3d(
	const float *pv1,
	const float *pv2
);

void
msNormalizeDataPositive(
	float *pfVec,
	int iLength
);

void
interpolate_discrete_3D_point(
	FEATUREIO &fioC,	// Center image: for interpolating feature geometry
	int ix, int iy, int iz,
	float &fx, float &fy, float &fz
);

double
interpolate_extremum_quadratic(
	double x0, // Three coordinates x
	double x1,
	double x2,
	double fx0, // Three functions of coordinates f(x)
	double fx1,
	double fx2
);

int
generateFeature3D(
	Feature3D &feat3D, // Feature geometrical information
	FEATUREIO &fioSample, // Sub-image sample
	FEATUREIO &fioImg, // Original image
	FEATUREIO &fioDx, // Derivative images
	FEATUREIO &fioDy,
	FEATUREIO &fioDz,
	vector<Feature3D> &vecFeats,
	float fEigThres = -1,// Threshold on eigenvalues, discard if
	int bReorientedFeatures = 1 // Wether to reorient features or not, default yes (1)
);

static int
regFindFEATUREIOValleys(
	LOCATION_VALUE_XYZ_ARRAY &lvaPeaks,
	FEATUREIO &fio,
	EXTREMA_FUNCTION pCallback = 0,
	EXTREMA_STRUCT *pES = 0
);

static int
regFindFEATUREIOPeaks(
	LOCATION_VALUE_XYZ_ARRAY &lvaPeaks,
	FEATUREIO &fio,
	EXTREMA_FUNCTION pCallback = 0,
	EXTREMA_STRUCT *pES = 0
);

static int
regFindFEATUREIO_interleave(
	LOCATION_VALUE_XYZ_ARRAY &lvaPeaks,
	LOCATION_VALUE_XYZ_ARRAY &lvaValleys,
	FEATUREIO &fio,
	EXTREMA_FUNCTION peakCallback = 0,
	EXTREMA_FUNCTION valleyCallback = 0,
	EXTREMA_STRUCT *pES = 0
);

static int
regFindFEATUREIO(
	LOCATION_VALUE_XYZ_ARRAY &lvaPeaks,
	LOCATION_VALUE_XYZ_ARRAY &lvaValleys,
	FEATUREIO &fio,
	EXTREMA_FUNCTION peakCallback = 0,
	EXTREMA_FUNCTION valleyCallback = 0,
	EXTREMA_STRUCT *pES = 0
);

static int
regFindFEATUREIO_NO_CALLBACK(
	LOCATION_VALUE_XYZ_ARRAY &lvaPeaks,
	LOCATION_VALUE_XYZ_ARRAY &lvaValleys,
	FEATUREIO &fio,
	EXTREMA_STRUCT *pES = 0
);


int
valleyFunction4D(
	EXTREMA_STRUCT *pES
);

int
peakFunction4D(
	EXTREMA_STRUCT *pES
);

double finddet(double a1, double a2, double a3, double b1, double b2, double b3, double c1, double c2, double c3);

int
determineOrientation3D(
	Feature3D &feat3D
);

int
sampleImage3D(
	Feature3D &feat3D,
	FEATUREIO &fioSample, // 5x5x5 = 125 feats?,
	FEATUREIO &fioImg, // Original image
	FEATUREIO &fioDx, // Derivative images
	FEATUREIO &fioDy,
	FEATUREIO &fioDz
);

int
determineCanonicalOrientation3D(
	Feature3D &feat3D,
	float *pfOri, // Array to store iMaxOri 3D rotation arrays. Each rotation array is
				  //			encoded as three 3D unit vectors
	int iMaxOri // Number of 6-float values available in pfOri
);

void
vec3D_cross_3d(
	float *pv1,
	float *pv2,
	float *pvCross
);

void
similarity_transform_invert(
	float *pfCenter0,
	float *pfCenter1,
	float *rot01,
	float &fScaleDiff
);

int
similarity_transform_3point(
	float *pfP0,
	float *pfP1,

	float *pfCenter0,
	float *pfCenter1,
	float *rot01,
	float fScaleDiff
);

void
vec3D_diff_3d(
	float *pv1,
	float *pv2,
	float *pv12
);

template <class T, class DIV>
void
transpose_3x3(
	T mat_in[3][3],
	T mat_trans[3][3]
)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			mat_trans[i][j] = mat_in[j][i];
		}
	}
}

void
vec3D_summ_3d(
	float *pv1,
	float *pv2,
	float *pv12
);

int
_sortAscendingMVNature(
	const void *pf1,
	const void *pf2
);

#endif
