#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <list>
#include <algorithm>

#include "MultiScale.h"
#include "GaussBlur3D.h"
#include "SIFT_cuda_Tools.cuh"
#include "LocationValue.h"
#include "PpImageFloatOutput.h"
#include "PpImage.h"
#include "SVD.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include <random>
#include <complex>

#include <iomanip>
#include <string>
#include <map>


// Most experiments use 1.0f;
// 0.5 - many false matches
// 1.5 - not bad, interestingly there are different correct matches
// 1.2 - not great
float fBlurGradOriHist = 0.5f;

// Most experiments use 0.5
float fHist2ndPeakThreshold = 0.5f;

#define BLUR_PRECISION 0.01f
#define PI 3.1415926536

//
// One octave is a doubling in scale, i.e. blur sigma.
// 3 blurs per octave is optimal, and an additional 3 are needed
// to perform DOG selection.
//
#define BLURS_PER_OCTAVE	3
#define BLURS_EXTRA			3
#define BLURS_TOTAL			(BLURS_PER_OCTAVE+BLURS_EXTRA)

Feature3DInfo::Feature3DInfo(
)
{
	std::memset(m_pfPC, 0, sizeof(m_pfPC));
	ZeroData();
}

void
Feature3DInfo::ZeroData(
)
{
	for (int i = 0; i < Feature3DInfo::FEATURE_3D_PCS; i++)
	{
		m_pfPC[i] = 0;
	}

	for (int i = 0; i < 3; i++)
	{
		eigs[i] = 0;
		for (int j = 0; j < 3; j++)
		{
			ori[i][j] = 0;
		}
	}

	m_uiInfo = 0;

	x = 0;
	y = 0;
	z = 0;
	scale = 0;
}

void
Feature3DInfo::SimilarityTransform(
	float *pfMat4x4
)
{
	float pfP0[4];
	float pfP1[4];

	// Transform point location
	pfP0[0] = x; pfP0[1] = y; pfP0[2] = z; pfP0[3] = 1;
	mult_4x4_vector(pfMat4x4, pfP0, pfP1);
	x = pfP1[0]; y = pfP1[1]; z = pfP1[2];

	float fScaleSum = 0;
	float pfScale[3];
	float pfRot3x3[3][3];

	for (int i = 0; i < 3; i++)
	{
		// Figure out scaling - should be isotropic
		pfScale[i] = vec3D_mag(&pfMat4x4[4 * i]);
		fScaleSum += pfScale[i];

		// Create rotation matrix - should be able to rescale
		memcpy(&pfRot3x3[i][0], &pfMat4x4[4 * i], 3 * sizeof(float));
		vec3D_norm_3d(&pfRot3x3[i][0]);
	}
	fScaleSum /= 3;

	// Transform scale
	scale *= fScaleSum;

	// Transform orientation/rotation matrix
	float pfOri[3][3];
	float pfOriOut[3][3];
	transpose_3x3<float, float>(ori, pfOri);
	mult_3x3_matrix<float, float>(pfRot3x3, pfOri, pfOriOut);
	transpose_3x3<float, float>(pfOriOut, ori);
}

void
Feature3D::NormalizeData(
	float *pfMean,
	float *pfVarr,
	float *pfMin,
	float *pfMax
)
{
	float fSum = 0;
	float fMax = data_zyx[0][0][0], fMin = data_zyx[0][0][0];

	for (int zz = 0; zz < Feature3D::FEATURE_3D_DIM; zz++)
	{
		for (int yy = 0; yy < Feature3D::FEATURE_3D_DIM; yy++)
		{
			for (int xx = 0; xx < Feature3D::FEATURE_3D_DIM; xx++)
			{
				fSum += data_zyx[zz][yy][xx];
				if (data_zyx[zz][yy][xx] < fMin)
				{
					fMin = data_zyx[zz][yy][xx];
				}
				if (data_zyx[zz][yy][xx] > fMax)
				{
					fMax = data_zyx[zz][yy][xx];
				}
			}
		}
	}
	float fMean = fSum / (Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM);
	float fSumSqr = 0;

	for (int zz = 0; zz < Feature3D::FEATURE_3D_DIM; zz++)
	{
		for (int yy = 0; yy < Feature3D::FEATURE_3D_DIM; yy++)
		{
			for (int xx = 0; xx < Feature3D::FEATURE_3D_DIM; xx++)
			{
				data_zyx[zz][yy][xx] -= fMean;
				fSumSqr += data_zyx[zz][yy][xx] * data_zyx[zz][yy][xx];
			}
		}
	}

	if (pfMean)
	{
		*pfMean = fMean;
	}
	if (pfVarr)
	{
		*pfVarr = fSumSqr;
	}

	if (pfMin)
	{
		*pfMin = fMin;
	}
	if (pfMax)
	{
		*pfMax = fMax;
	}

	float fDiv = 1.0f / (float)sqrt(fSumSqr);
	fSum = 0;
	fSumSqr = 0;

	for (int zz = 0; zz < Feature3D::FEATURE_3D_DIM; zz++)
	{
		for (int yy = 0; yy < Feature3D::FEATURE_3D_DIM; yy++)
		{
			for (int xx = 0; xx < Feature3D::FEATURE_3D_DIM; xx++)
			{
				data_zyx[zz][yy][xx] *= fDiv;
				fSum += data_zyx[zz][yy][xx];
				fSumSqr += data_zyx[zz][yy][xx] * data_zyx[zz][yy][xx];
			}
		}
	}
}

void
Feature3DInfo::NormalizeDataRankedPCs(
)
{
	COMP_MATCH_VALUE mvScan[PC_ARRAY_SIZE];

	// Load up vector
	int k = 0;
	float fMin = 100000;

	for (int zz = 0; zz < PC_ARRAY_SIZE; zz++)
	{
		mvScan[k].iIndex1 = zz;
		mvScan[k].iIndex2 = 0;
		mvScan[k].fValue = m_pfPC[zz];
		k++;
	}

	// Sort descending
	qsort((void*)(mvScan), PC_ARRAY_SIZE, sizeof(COMP_MATCH_VALUE), _sortAscendingMVNature);

	// Asign indices
	for (k = 0; k < PC_ARRAY_SIZE; k++)
	{
		m_pfPC[mvScan[k].iIndex1] = k;
	}
}


int
msGeneratePyramidDOG3D_efficient(
	FEATUREIO	&fioImg,
	vector<Feature3D> &vecFeats,
	int best_device_id,
	float fInitialImageScale,
	int bDense,
	int bOutput,
	float fEigThres
)
{
	int iScaleCount = 4;
	char pcFileName[300];
	PpImage ppImgOut;
	PpImage ppImgOut_After_Gauss;
	PpImage ppImgOutMem;
	ppImgOutMem.Initialize(1, fioImg.x*fioImg.y*fioImg.z, fioImg.x*fioImg.y*fioImg.z * sizeof(float), sizeof(float) * 8);

	// DoG extrema candidates
	vector<LOCATION_VALUE_XYZ> vecLVAMin;
	vector<LOCATION_VALUE_XYZ> vecLVAMax;
	vecLVAMin.resize(fioImg.x*fioImg.y);
	vecLVAMax.resize(fioImg.x*fioImg.y);

	int i;

	// Holders for DoG values above & below extrema
	vector<float> vecMinH, vecMinL, vecMaxH, vecMaxL;
	vecMinH.resize(fioImg.x*fioImg.y);
	vecMinL.resize(fioImg.x*fioImg.y);
	vecMaxH.resize(fioImg.x*fioImg.y);
	vecMaxL.resize(fioImg.x*fioImg.y);

	// Should be able to do everyting with 4 extra images
	// Blur pyramid is a 3D FEATUREIO
	FEATUREIO fioBlurs[5];
	for (i = 0; i < 5; i++)
	{
		fioBlurs[i] = fioImg;
		fioAllocate(fioBlurs[i]);
	}
	// Half dimension image for saving 2X octave
	FEATUREIO fioSaveHalf = fioImg;
	fioSaveHalf.x /= 2;
	fioSaveHalf.y /= 2;
	fioSaveHalf.z /= 2;
	fioAllocate(fioSaveHalf);

	//
	// Here, the initial blur sigma is calculated along with the subsequent
	// sigma increments.
	//
	float fSigmaInit = 0.5; // initial blurring: image resolution
	if (fInitialImageScale > 0) fSigmaInit /= fInitialImageScale;
	float fSigma = 1.6f; // desired initial blurring
	float fSigmaFactor = (float)pow(2.0, 1.0 / (double)BLURS_PER_OCTAVE);

	// Start initial blur: blur_2^2 = blur_1^2 + fSigmaExtra^2
	float fSigmaExtra = sqrt(fSigma*fSigma - fSigmaInit*fSigmaInit);

	auto start = high_resolution_clock::now();

	gb3d_blur3d(fioImg, fioBlurs[1], fioBlurs[0], fSigmaExtra, BLUR_PRECISION, best_device_id);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "\n#" << duration.count() << endl;

	// Output initial image
	ppImgOut.InitializeSubImage(
		fioImg.y,
		fioImg.x,
		fioImg.x * sizeof(float), sizeof(float) * 8,
		(unsigned char*)ppImgOutMem.ImageRow(0)
	);
	fioFeatureSliceXY(fioImg, fioImg.z / 2, 0, (float*)ppImgOut.ImageRow(0));
	sprintf(pcFileName, "image.pgm");
	if (bOutput) output_float(ppImgOut, pcFileName);


	float fScale = 1;

	float pfBlurSigmas[BLURS_TOTAL];

	// Temporary image is the original image - a little distructive ...
	FEATUREIO fioTemp = fioImg;

	// Initialize only the size of these images, memory pointers will be modified later
	FEATUREIO fioG0 = fioImg;
	FEATUREIO fioG1 = fioImg;
	FEATUREIO fioG2 = fioImg;
	FEATUREIO fioD0 = fioImg;
	FEATUREIO fioD1 = fioImg;
	FEATUREIO fioSumOfSign = fioImg;

	LOCATION_VALUE_XYZ_ARRAY lvaMaxima, lvaMinima;
	LOCATION_VALUE_XYZ_ARRAY lvaMax2, lvaMin2;
	lvaMax2.iCount=0;
	lvaMin2.iCount=0;

	int iFeatDiff = 1;
	for (int i = 0; 1; i++)
	{
		//printf("Scale %d: blur...", i);

		fSigma = 1.6f; // desired initial blurring

		pfBlurSigmas[0] = fSigma;

		// Init pointers to image memory
		fioG0.pfVectors = fioBlurs[0].pfVectors;
		fioG1.pfVectors = fioBlurs[1].pfVectors;
		fioG2.pfVectors = fioBlurs[2].pfVectors;
		fioD0.pfVectors = fioBlurs[3].pfVectors;
		fioSumOfSign.pfVectors = fioBlurs[4].pfVectors;


		fioG0.d_pfVectors = fioBlurs[0].d_pfVectors;
		fioG1.d_pfVectors = fioBlurs[1].d_pfVectors;
		fioG2.d_pfVectors = fioBlurs[2].d_pfVectors;
		fioD0.d_pfVectors = fioBlurs[3].d_pfVectors;
		fioSumOfSign.d_pfVectors = fioBlurs[4].d_pfVectors;

		if (fioG0.x <= 2 || fioG0.y <= 2 || fioG0.z <= 2)
			goto finish_up; // Quit if too small

		int iFirstFeat = vecFeats.size();

		// Generate blurs in octave
		for (int j = 1; j < BLURS_TOTAL; j++)
		{
			// How much extra blur required to raise fSigma-blurred image
			// to next blur level
			float fSigmaExtra = fSigma*sqrt(fSigmaFactor*fSigmaFactor - 1.0f);
			if (j == 1)
			{
				start = high_resolution_clock::now();
				int iReturn = gb3d_blur3d(fioG0, fioTemp, fioG1, fSigmaExtra, BLUR_PRECISION, best_device_id);
				ppImgOut.InitializeSubImage(
					fioG1.y,
					fioG1.x,
					fioG1.x * sizeof(float), sizeof(float) * 8,
					(unsigned char*)ppImgOutMem.ImageRow(0)
				);
				fioFeatureSliceXY(fioG1, fioG1.z / 2, 0, (float*)ppImgOut.ImageRow(0));
				sprintf(pcFileName, "image.pgm");
				if (i==0) {
					output_float(ppImgOut, pcFileName);
				}

				stop = high_resolution_clock::now();
				duration = duration_cast<microseconds>(stop - start);
				cout << "\n#" << duration.count() << endl;
				if (iReturn == 0)
					goto finish_up;

				// Initialize Dog
				start = high_resolution_clock::now();
				fioMultSum_interleave(fioG0, fioG1, fioD0, -1.0f, best_device_id);
				stop = high_resolution_clock::now();
				duration = duration_cast<microseconds>(stop - start);
				cout << "\n#" << duration.count() << endl;
			}
			else
			{
				start = high_resolution_clock::now();
				int iReturn = gb3d_blur3d(fioG1, fioTemp, fioG2, fSigmaExtra, BLUR_PRECISION, best_device_id);
				stop = high_resolution_clock::now();
				duration = duration_cast<microseconds>(stop - start);
				cout << "\n#" << duration.count() << endl;
				if (iReturn == 0)
					goto finish_up;

				if (j == BLURS_PER_OCTAVE)
				{
					// Save 2X octave image for next round
					start = high_resolution_clock::now();
					Subsample_interleave(fioG2, fioSaveHalf, best_device_id);
					stop = high_resolution_clock::now();
					duration = duration_cast<microseconds>(stop - start);
					cout << "\n#" << duration.count() << endl;
				}

				if (j >= 3)
				{
					// A list exists, verify and compute features

					// Verify Maxima from previous level
					int iCurrTop = 0;
					for (int k = 0; k < lvaMaxima.iCount; k++)
					{
						if (validateDifferencePeak3D(vecLVAMax[k], fioG1, fioG2))
						{
							// Save this value (overwrite)
							vecMaxH[iCurrTop] = vecMaxH[k];
							vecMaxL[iCurrTop] = fioGetPixel(fioG1, vecLVAMax[k].x, vecLVAMax[k].y, vecLVAMax[k].z) - fioGetPixel(fioG2, vecLVAMax[k].x, vecLVAMax[k].y, vecLVAMax[k].z);
							vecLVAMax[iCurrTop] = vecLVAMax[k];
							iCurrTop++;
						}
					}
					lvaMaxima.iCount = iCurrTop;


					// Verify Minima from previous level
					iCurrTop = 0;
					//printf("\t\t%d\n", lvaMinima.iCount);
					for (int k = 0; k < lvaMinima.iCount; k++)
					{
						if (validateDifferenceValley3D(vecLVAMin[k], fioG1, fioG2))
						{
							// Save this value (overwrite)
							vecMinH[iCurrTop] = vecMinH[k];
							vecMinL[iCurrTop] = fioGetPixel(fioG1, vecLVAMin[k].x, vecLVAMin[k].y, vecLVAMin[k].z) - fioGetPixel(fioG2, vecLVAMin[k].x, vecLVAMin[k].y, vecLVAMin[k].z);
							vecLVAMin[iCurrTop] = vecLVAMin[k];
							iCurrTop++;
						}
					}
					lvaMinima.iCount = iCurrTop;
					lvaMaxima.plvz = &(vecLVAMax[0]);
					lvaMinima.plvz = &(vecLVAMin[0]);

					// Generate features for points found
					// Have to send in feature values above/below for interpolation
					generateFeatures3D_efficient(
						lvaMinima, lvaMaxima,
						vecMinH, vecMinL, vecMaxH, vecMaxL,
						fioD0,
						pfBlurSigmas[j - 3], pfBlurSigmas[j - 2], pfBlurSigmas[j - 1],
						fioG0,
						pfBlurSigmas[j + 1],
						vecFeats, fEigThres
					);
				}

				if (j < BLURS_TOTAL - 1)
				{
					// Lists and fioG0 are now free

					// Difference: fioD1 = fioG1 - fioG2
					fioD1.pfVectors = fioG0.pfVectors;
					fioD1.d_pfVectors = fioG0.d_pfVectors;
					start = high_resolution_clock::now();
					fioMultSum_interleave(fioG1, fioG2, fioD1, -1.0f, best_device_id);
					stop = high_resolution_clock::now();
					duration = duration_cast<microseconds>(stop - start);
					cout << "\n#" << duration.count() << endl;
					// Identify peak candidates
					lvaMaxima.plvz = &(vecLVAMax[0]);
					lvaMinima.plvz = &(vecLVAMin[0]);
					start = high_resolution_clock::now();
					detectExtrema4D_test_interleave(
						fioD0, fioD1, fioSumOfSign,
						lvaMinima,
						lvaMaxima,
						best_device_id
					);
					stop = high_resolution_clock::now();
					duration = duration_cast<microseconds>(stop - start);
					cout << "\n#" << duration.count() << endl;

					// Save values for scale below Maxima
					for (int k = 0; k < lvaMaxima.iCount; k++)
					{
						vecMaxH[k] = fioGetPixel(fioD0, vecLVAMax[k].x, vecLVAMax[k].y, vecLVAMax[k].z);
					}

					// Save values for scale below Minima
					for (int k = 0; k < lvaMinima.iCount; k++)
					{
						vecMinH[k] = fioGetPixel(fioD0, vecLVAMin[k].x, vecLVAMin[k].y, vecLVAMin[k].z);
					}

					// fioD0 is now free
					float *pfG2Temp = fioG2.pfVectors;
					fioG2.pfVectors = fioD0.pfVectors;
					fioD0.pfVectors = fioD1.pfVectors;
					fioG0.pfVectors = fioG1.pfVectors;
					fioG1.pfVectors = pfG2Temp;

					int iDataSizeFloat = fioG2.x*fioG2.y*fioG2.z*fioG2.t*fioG2.iFeaturesPerVector*sizeof(float);


					float *d_pfG2Temp = fioG2.d_pfVectors;
					fioG2.d_pfVectors = fioD0.d_pfVectors;
					fioD0.d_pfVectors = fioD1.d_pfVectors;
					fioG0.d_pfVectors = fioG1.d_pfVectors;
					fioG1.d_pfVectors = d_pfG2Temp;
				}
			}

			fSigma *= fSigmaFactor;
			pfBlurSigmas[j] = fSigma;
		}

		// Update geometry to image space
		float fFactor = fScale;
		float fAddend = 0;
		for (int iFU = iFirstFeat; iFU < vecFeats.size(); iFU++)
		{
			// Update feature scale to actual
			vecFeats[iFU].scale *= fFactor;

			// Should also update location...
			vecFeats[iFU].x = vecFeats[iFU].x*fFactor + fAddend;
			vecFeats[iFU].y = vecFeats[iFU].y*fFactor + fAddend;
			vecFeats[iFU].z = vecFeats[iFU].z*fFactor + fAddend;
		}
		fScale *= 2.0f;

		// Initialize saved image
		fioBlurs[0].x /= 2;
		fioBlurs[0].y /= 2;
		fioBlurs[0].z /= 2;
		fioCopy(fioBlurs[0], fioSaveHalf);
		fioSaveHalf.x /= 2;
		fioSaveHalf.y /= 2;
		fioSaveHalf.z /= 2;

		fioTemp.x = fioG0.x = fioG1.x = fioG2.x = fioD0.x = fioD1.x = fioSumOfSign.x = fioBlurs[0].x;
		fioTemp.y = fioG0.y = fioG1.y = fioG2.y = fioD0.y = fioD1.y = fioSumOfSign.y = fioBlurs[0].y;
		fioTemp.z = fioG0.z = fioG1.z = fioG2.z = fioD0.z = fioD1.z = fioSumOfSign.z = fioBlurs[0].z;

		printf("done.\n");
	}

finish_up:

	for (i = 0; i < 5; i++)
	{
		fioDelete(fioBlurs[i]);
	}
	fioDelete(fioSaveHalf);

	return 1;
}

void display(int start, int nb)
{
	for (int i = start; i < start + nb; ++i)
		std::cout << i << ",";
}

//
// msResampleFeaturesGradientOrientationHistogram()
//
// Resample as gradient orientation histograms.
//
int
msResampleFeaturesGradientOrientationHistogram(
	Feature3D &feat3D
)
{

	FEATUREIO fioImg;
	fioImg.x = fioImg.z = fioImg.y = Feature3D::FEATURE_3D_DIM;
	fioImg.pfVectors = &(feat3D.data_zyx[0][0][0]);
	fioImg.pfMeans = 0;
	fioImg.pfVarrs = 0;
	fioImg.iFeaturesPerVector = 1;
	fioImg.t = 1;
	// 1st derivative images - these could be passed in from the outside.
	FEATUREIO fioDx = fioImg;
	FEATUREIO fioDy = fioImg;
	FEATUREIO fioDz = fioImg;
	Feature3D fdx;
	Feature3D fdy;
	Feature3D fdz;
	fioDx.pfVectors = &(fdx.data_zyx[0][0][0]);
	fioDy.pfVectors = &(fdy.data_zyx[0][0][0]);
	fioDz.pfVectors = &(fdz.data_zyx[0][0][0]);

	fioGenerateEdgeImages3D(fioImg, fioDx, fioDy, fioDz);

	float fRadius = fioImg.x / 2;
	float fRadiusSqr = (fioImg.x / 2)*(fioImg.x / 2);

	int iSampleCount = 0;

	// Consider 8 orientation angles
#define GRAD_ORI_ORIBINS 8
	float pfOriAngles[GRAD_ORI_ORIBINS][3] =
	{
		{ 1,	 1,	 1 },
		{ 1,	 1,	-1 },
		{ 1,	-1,	 1 },
		{ 1,	-1,	-1 },
		{ -1,	 1,	 1 },
		{ -1,	 1,	-1 },
		{ -1,	-1,	 1 },
		{ -1,	-1,	-1 },
	};

	// Reference to grad orientation info
#define GRAD_ORI_SPACEBINS 2
	float fBinSize = Feature3D::FEATURE_3D_DIM / (float)GRAD_ORI_SPACEBINS;
	FEATUREIO fioImgGradOri;
	fioImgGradOri.x = fioImgGradOri.z = fioImgGradOri.y = 2;
	assert(Feature3DInfo::FEATURE_3D_PCS >= 64);
	fioImgGradOri.pfVectors = &(feat3D.m_pfPC[0]);
	fioImgGradOri.pfMeans = 0;
	fioImgGradOri.pfVarrs = 0;
	fioImgGradOri.iFeaturesPerVector = GRAD_ORI_ORIBINS;
	fioImgGradOri.t = 1;
	for (int zz = 0; zz < fioImg.z; zz++)
	{
		float fZCoord = (int)(zz / fBinSize) + 0.5f;
		if ((int)((zz + 0) / fBinSize) != (int)((zz + 1) / fBinSize))
		{
			float fP0 = ((zz + 0) / fBinSize);
			float fP1 = ((zz + 1) / fBinSize);
			fZCoord = (fP0 + fP1) / 2.0f;
		}

		for (int yy = 0; yy < fioImg.y; yy++)
		{
			float fYCoord = (int)(yy / fBinSize) + 0.5f;
			if ((int)((yy + 0) / fBinSize) != (int)((yy + 1) / fBinSize))
			{
				float fP0 = ((yy + 0) / fBinSize);
				float fP1 = ((yy + 1) / fBinSize);
				fYCoord = (fP0 + fP1) / 2.0f;
			}

			for (int xx = 0; xx < fioImg.x; xx++)
			{

				float fXCoord = (int)(xx / fBinSize) + 0.5f;
				if ((int)((xx + 0) / fBinSize) != (int)((xx + 1) / fBinSize))
				{
					float fIntPart;
					int iNext = (int)((xx + 1) / fBinSize);

					float fP0 = ((xx + 0) / fBinSize);
					float fP1 = ((xx + 1) / fBinSize);
					fXCoord = (fP0 + fP1) / 2.0f;
				}

				float dz = zz - fioImg.z / 2;
				float dy = yy - fioImg.y / 2;
				float dx = xx - fioImg.x / 2;

				// Keep this sample
				float pfEdge[3];
				pfEdge[0] = fioGetPixel(fioDx, xx, yy, zz);
				pfEdge[1] = fioGetPixel(fioDy, xx, yy, zz);
				pfEdge[2] = fioGetPixel(fioDz, xx, yy, zz);
				float fEdgeMag = vec3D_mag(&pfEdge[0]);
				if (fEdgeMag > 0)
				{
					vec3D_norm_3d(pfEdge);

					// Find max orientation bin: this could be made more efficient ...
					int iMaxDotIndex = 0;
					float fMaxDot = vec3D_dot_3d(&pfOriAngles[iMaxDotIndex][0], &pfEdge[0]);
					for (int k = 1; k < 8; k++)
					{
						float fDot = vec3D_dot_3d(&pfOriAngles[k][0], &pfEdge[0]);
						if (fDot > fMaxDot)
						{
							fMaxDot = fDot;
							iMaxDotIndex = k;
						}
					}

					fioIncPixelTrilinearInterp(fioImgGradOri, fXCoord, fYCoord, fZCoord, iMaxDotIndex, fEdgeMag);
				}
			}
		}
	}

	// Normalize the PC vector
	msNormalizeDataPositive(&(feat3D.m_pfPC[0]), Feature3DInfo::FEATURE_3D_PCS);

	return 0;
}

//
// msGenerateBRIEFindex()
//
// Generate pseudo random index (depending to algorithms) for BRIEF Algorithm
//
//

int
msGenerateBRIEFindex(
	vector<LOCATION_VALUE_XYZ> &RandomIndexX,
	vector<LOCATION_VALUE_XYZ> &RandomIndexY,
	int iCount,
	FEATUREIO fio,
	int method
)
{
	int random_seed=5;
	int random_seed2=8;
	RandomIndexX.resize(iCount);
	RandomIndexY.resize(iCount);
	int i =0;
	int *iListPos=NULL;
	iListPos=new int[Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM];
	for(int z=0; z<Feature3D::FEATURE_3D_DIM; z++){
		for(int y=0; y<Feature3D::FEATURE_3D_DIM; y++){
			for(int x=0; x<Feature3D::FEATURE_3D_DIM; x++){
				iListPos[x+y*Feature3D::FEATURE_3D_DIM+z*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM]=0;
			}
		}
	}
	if (method==0) {
		int xyz0[192]={4,6,2,2,2,2,4,3,8,7,3,2,2,6,3,3,5,8,6,7,5,5,7,4,6,6,3,2,6,8,2,7,2,6,6,7,7,8,8,6,3,2,4,5,5,4,7,7,5,7,4,3,7,2,2,3,8,3,2,4,3,5,4,3,4,2,6,6,5,8,2,3,3,4,7,8,3,2,2,7,3,5,4,5,6,5,6,7,6,8,4,8,4,5,8,5,6,3,6,5,3,7,6,3,8,6,8,2,8,2,8,3,2,3,3,5,3,7,8,3,4,4,5,5,3,2,8,7,6,5,3,6,4,2,4,2,7,5,4,6,7,3,5,4,3,5,2,6,3,2,8,4,4,6,5,4,8,7,2,8,6,5,2,7,5,7,4,2,5,7,4,7,7,4,8,8,2,8,3,4,6,7,5,8,2,4,6,3,8,6,5,4};
		int xyz1[192]={5,2,3,7,5,8,7,5,6,5,6,3,2,7,4,6,2,8,4,6,6,3,5,7,7,4,3,3,4,8,8,5,3,4,2,6,8,3,3,3,7,8,6,2,6,6,2,5,2,7,8,6,2,7,4,3,8,4,7,7,3,3,8,2,5,2,7,2,4,5,8,3,5,6,3,2,8,2,4,6,7,3,2,4,4,7,4,4,8,8,5,8,2,8,8,5,3,3,5,6,7,4,8,4,8,7,4,7,3,4,6,7,5,2,8,7,6,5,8,7,8,7,8,6,8,4,8,4,5,7,4,8,2,3,8,2,5,4,3,2,8,8,7,3,5,7,4,5,4,6,6,7,7,8,6,8,4,2,6,7,5,4,2,8,8,6,5,8,4,4,4,6,6,4,5,3,4,5,4,4,8,4,3,4,6,5,8,7,7,2,2,3};

		// Only for futur : generate 64 pair of random number under uniform distribution just put random number in randomseed
		// Fully random index algorithm folowing uniform distribution

		//std::random_device rd; // obtain a random number from hardware
    /*std::mt19937 eng(random_seed); // seed the generator
		cout << ceil(3*Feature3D::FEATURE_3D_DIM/4);
    std::uniform_int_distribution<> distr(ceil(Feature3D::FEATURE_3D_DIM/4), ceil(3*Feature3D::FEATURE_3D_DIM/4)); // define the range

		std::map<int, int> hist;
		for (int n = 0; n < 100000; ++n) {
			++hist[std::round(distr(eng))];
		}
		std::cout << "\nThis is uniform distribution representation: \n\n";
		for (auto p : hist) {
			std::cout << std::fixed << std::setprecision(1) << std::setw(2)
								<< p.first << ' ' << std::string(p.second/200, '*') << '\n';
		}
		*/
		for(int i=0; i<64; i++){
			RandomIndexX[i].x = xyz0[i*3];
			RandomIndexX[i].y = xyz0[i*3+1];
			RandomIndexX[i].z = xyz0[i*3+2];
			RandomIndexY[i].x = xyz1[i*3];
			RandomIndexY[i].y = xyz1[i*3+1];
			RandomIndexY[i].z = xyz1[i*3+2];
		}
	}

	else if(method==1){
		int xyz0[192]={5,4,4,6,5,5,3,8,5,5,6,3,5,6,5,6,3,4,3,4,5,4,5,4,5,5,5,5,6,5,5,5,5,3,5,7,3,5,5,5,6,6,5,3,6,5,5,5,4,5,5,5,3,5,4,4,6,6,4,3,5,3,3,3,6,6,4,4,5,5,5,5,4,4,5,6,5,4,4,4,4,3,4,4,6,3,2,5,4,4,5,4,3,6,7,5,3,5,4,5,5,4,5,6,3,5,6,5,5,6,5,5,7,6,4,4,6,6,4,4,4,5,2,5,4,5,2,5,5,5,2,6,3,3,5,4,7,5,4,5,3,5,4,6,4,4,3,4,5,4,6,3,4,5,5,6,4,3,4,6,4,4,6,5,4,4,5,5,5,5,4,4,3,7,7,3,6,6,5,7,4,6,2,4,2,5,6,3,3,6,5,6};

		int xyz1[192]={4,4,2,4,4,4,5,6,4,5,5,5,4,6,6,4,4,5,4,5,5,4,6,4,4,2,7,7,5,3,5,4,5,4,5,4,2,3,5,4,5,5,4,5,5,4,6,5,4,4,6,4,5,5,3,6,4,6,4,4,7,4,5,4,4,2,5,4,6,4,3,5,3,4,7,5,2,4,4,6,3,4,6,5,6,4,4,5,5,3,4,5,4,5,5,5,4,5,5,4,5,4,5,3,4,6,4,5,3,6,5,4,4,6,4,7,4,4,3,6,4,3,7,4,5,6,2,3,6,5,5,5,5,4,4,5,3,4,6,4,5,5,4,2,4,4,4,6,4,6,6,3,6,5,5,3,3,5,5,3,5,3,4,2,3,6,2,4,5,4,7,3,4,3,3,5,4,3,5,4,4,4,6,3,5,4,3,5,7,5,4,4};
		/*
		// isotropic Gaussian distribution mean centered
    std::mt19937 eng(random_seed); // seed the generator
    std::normal_distribution<> distr{int(Feature3D::FEATURE_3D_DIM/2), (((Feature3D::FEATURE_3D_DIM/2))*((Feature3D::FEATURE_3D_DIM/2)))/25}; // define the range

		std::cout << "\nThis is normal distribution representation: \n\n";
		std::map<int, int> hist;
	  for (int n = 0; n < 100000; ++n) {
	    ++hist[std::round(distr(eng))];
	  }

	  for (auto p : hist) {
	    std::cout << std::fixed << std::setprecision(1) << std::setw(2)
	              << p.first << ' ' << std::string(p.second/200, '*') << '\n';
	  }*/

		for(int i=0; i<64; i++){
			RandomIndexX[i].x = xyz0[i*3];
			RandomIndexX[i].y = xyz0[i*3+1];
			RandomIndexX[i].z = xyz0[i*3+2];
			RandomIndexY[i].x = xyz1[i*3];
			RandomIndexY[i].y = xyz1[i*3+1];
			RandomIndexY[i].z = xyz1[i*3+2];
		}
	}
	else if(method==2){

		int xyz0[192]={5,4,4,4,4,2,6,5,5,4,4,4,3,8,5,5,6,3,5,5,5,5,6,5,4,6,6,6,3,4,4,4,5,3,4,5,4,5,5,4,2,7,7,5,3,5,4,5,3,5,7,3,5,5,2,3,5,5,6,6,4,6,5,4,4,6,5,3,5,6,4,3,6,4,4,5,3,3,3,6,6,5,2,4,4,6,3,6,3,2,3,5,4,5,3,4,3,6,5,4,3,6,4,5,2,4,3,7,2,3,6,5,2,6,3,3,5,6,3,6,3,5,3,6,5,7,4,2,5,5,5,2,5,7,4,2,5,3,4,3,3,7,4,4,7,6,4,4,2,8,7,6,5,4,7,3,6,6,5,2,4,5,3,2,5,5,1,6,3,6,3,6,2,5,4,4,7,2,6,3,2,2,4,3,3,2,3,4,2,5,6,7};

		int xyz1[192]={6,5,3,4,5,3,7,4,6,4,3,2,4,7,5,3,5,1,5,4,7,6,8,4,4,5,6,5,2,5,4,6,4,0,4,3,3,4,4,2,1,7,8,6,4,4,1,6,1,3,7,2,3,3,1,3,6,1,6,6,4,7,6,4,3,5,4,2,3,6,4,5,6,3,3,5,1,3,1,6,7,4,1,4,3,5,2,4,2,1,2,5,4,5,2,3,3,3,3,4,2,6,3,4,3,3,3,6,1,2,5,4,2,4,1,4,6,7,3,6,2,4,3,6,5,6,4,0,6,6,5,1,4,7,2,1,5,3,4,2,2,7,3,3,6,4,2,4,1,9,7,7,5,2,7,1,7,5,5,1,5,4,1,3,3,4,0,5,1,6,3,5,3,2,3,3,7,2,5,1,1,0,4,1,3,1,0,3,1,6,5,9};

		// isotropic Gaussian distribution mean centered, for first position and centered on first position for second position
    /*std::mt19937 eng1(random_seed); // seed the generator
		float stddev = ((int(Feature3D::FEATURE_3D_DIM/2))*(int(Feature3D::FEATURE_3D_DIM/2)))/25;
    std::normal_distribution<> distr1{Feature3D::FEATURE_3D_DIM/2, stddev}; // define the range

    std::mt19937 eng2(random_seed2); // seed the generator
		std::map<int, int> hist;
		std::cout << "\nThis is normal distribution representation : \n\n";
		for (int n = 0; n < 100000; ++n) {
			++hist[std::round(distr1(eng1))];
		}
		for (auto p : hist) {
			std::cout << std::fixed << std::setprecision(1) << std::setw(2)
								<< p.first << ' ' << std::string(p.second/200, '*') << '\n';
		}
		while(i<iCount){

			RandomIndexX[i].x = distr1(eng1);
			RandomIndexX[i].y = distr1(eng1);
			RandomIndexX[i].z = distr1(eng1);
			std::normal_distribution<> distrX{RandomIndexX[i].x, stddev};
			std::normal_distribution<> distrY{RandomIndexX[i].y, stddev};
			std::normal_distribution<> distrZ{RandomIndexX[i].z, stddev};
			if (i==0) {
				std::cout << "\nThis is normal distribution representation (first point centered) : \n\n";
				printf("\nx = %d :\n", RandomIndexX[i].x);
				std::map<int, int> hist;
				for (int n = 0; n < 100000; ++n) {
					++hist[std::round(distrX(eng2))];
				}

				for (auto p : hist) {
					std::cout << std::fixed << std::setprecision(1) << std::setw(2)
										<< p.first << ' ' << std::string(p.second/200, '*') << '\n';
				}
				printf("\ny = %d :\n", RandomIndexX[i].y);
				hist.clear();
				for (int n = 0; n < 100000; ++n) {
					++hist[std::round(distrY(eng2))];
				}

				for (auto p : hist) {
					std::cout << std::fixed << std::setprecision(1) << std::setw(2)
										<< p.first << ' ' << std::string(p.second/200, '*') << '\n';
				}
				printf("\nz = %d :\n",RandomIndexX[i].z);
				hist.clear();
				for (int n = 0; n < 100000; ++n) {
					++hist[std::round(distrZ(eng2))];
				}

				for (auto p : hist) {
					std::cout << std::fixed << std::setprecision(1) << std::setw(2)
										<< p.first << ' ' << std::string(p.second/200, '*') << '\n';
				}
			}
			if (iListPos[RandomIndexX[i].x+RandomIndexX[i].y*Feature3D::FEATURE_3D_DIM+RandomIndexX[i].z*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM]!=0) {
				continue;
			}
			else{
				iListPos[RandomIndexX[i].x+RandomIndexX[i].y*Feature3D::FEATURE_3D_DIM+RandomIndexX[i].z*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM]+=1;
			}
			RandomIndexY[i].x = distrX(eng2);
			RandomIndexY[i].y = distrY(eng2);
			RandomIndexY[i].z = distrZ(eng2);
			if (iListPos[RandomIndexY[i].x+RandomIndexY[i].y*Feature3D::FEATURE_3D_DIM+RandomIndexY[i].z*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM]!=0) {
				continue;
			}
			else{
				iListPos[RandomIndexY[i].x+RandomIndexY[i].y*Feature3D::FEATURE_3D_DIM+RandomIndexY[i].z*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM]+=1;
				i++;
			}
		}*/
		for(int i=0; i<64; i++){
			RandomIndexX[i].x = xyz0[i*3];
			RandomIndexX[i].y = xyz0[i*3+1];
			RandomIndexX[i].z = xyz0[i*3+2];
			RandomIndexY[i].x = xyz1[i*3];
			RandomIndexY[i].y = xyz1[i*3+1];
			RandomIndexY[i].z = xyz1[i*3+2];
		}
	}
	else if(method==3){
		// isotropic Gaussian distribution for second position, first position to center
		/*
		std::random_device rd2; // obtain a random number from hardware
    std::mt19937 eng2(random_seed2); // seed the generator
		float stddev = ((int(Feature3D::FEATURE_3D_DIM/2))*(int(Feature3D::FEATURE_3D_DIM/2)))/25;
		std::normal_distribution<> distr{int(Feature3D::FEATURE_3D_DIM/2), stddev};
		while(i<iCount){
			RandomIndexX[i].x = int(Feature3D::FEATURE_3D_DIM/2);
			RandomIndexX[i].y = int(Feature3D::FEATURE_3D_DIM/2);
			RandomIndexX[i].z = int(Feature3D::FEATURE_3D_DIM/2);

			RandomIndexY[i].x = distr(eng2);
			RandomIndexY[i].y = distr(eng2);
			RandomIndexY[i].z = distr(eng2);

			if (iListPos[RandomIndexY[i].x+RandomIndexY[i].y*Feature3D::FEATURE_3D_DIM+RandomIndexY[i].z*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM]>0) {
				continue;
			}
			else{
				iListPos[RandomIndexX[i].x+RandomIndexX[i].y*Feature3D::FEATURE_3D_DIM+RandomIndexX[i].z*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM]+=1;
				iListPos[RandomIndexY[i].x+RandomIndexY[i].y*Feature3D::FEATURE_3D_DIM+RandomIndexY[i].z*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM]+=1;
				i++;

			}
		}*/
		int xyz0[3]={5,5,5};
		int xyz1[192]={6,4,6,3,4,6,5,4,6,4,6,4,6,3,4,4,6,2,5,5,4,5,3,4,6,5,4,4,5,4,4,4,4,5,4,5,3,5,4,3,3,4,6,7,5,6,4,7,4,4,6,5,4,4,4,3,4,5,6,4,5,3,7,5,4,3,2,5,5,3,4,4,4,5,6,5,6,3,4,3,2,4,6,3,3,4,3,4,4,3,5,3,5,4,4,5,1,6,5,4,5,5,5,6,6,5,4,2,5,5,6,5,7,4,3,5,3,4,3,7,3,7,5,3,6,4,6,4,4,6,3,5,6,4,5,5,7,5,2,4,3,7,6,5,7,4,6,6,5,5,4,5,3,4,3,5,5,5,3,5,3,3,4,6,5,6,6,6,6,6,5,4,2,4,6,6,3,3,5,5,7,3,4,4,4,2,4,6,6,5,6,5};
		for(int i=0; i<64; i++){
			RandomIndexX[i].x = xyz0[0];
			RandomIndexX[i].y = xyz0[1];
			RandomIndexX[i].z = xyz0[2];
			RandomIndexY[i].x = xyz1[i*3];
			RandomIndexY[i].y = xyz1[i*3+1];
			RandomIndexY[i].z = xyz1[i*3+2];
		}
}
	else if(method==4){
		// polar grid, first position to center
		/*std::mt19937 eng(random_seed); // seed the generator
		std::uniform_real_distribution<> magnitude{0.0, Feature3D::FEATURE_3D_DIM/2};
		std::uniform_real_distribution<> zenithAng{0.0, PI};
		std::uniform_real_distribution<> azimuthAng{0.0, 2*PI};

		while(i<iCount){
			int x, y, z;
			sphericalToCartesian(magnitude(eng), zenithAng(eng), azimuthAng(eng), x, y, z);
			RandomIndexX[i].x = int(Feature3D::FEATURE_3D_DIM/2);
			RandomIndexX[i].y = int(Feature3D::FEATURE_3D_DIM/2);
			RandomIndexX[i].z = int(Feature3D::FEATURE_3D_DIM/2);

			RandomIndexY[i].x = int(Feature3D::FEATURE_3D_DIM/2) + x;
			RandomIndexY[i].y = int(Feature3D::FEATURE_3D_DIM/2) + y;
			RandomIndexY[i].z = int(Feature3D::FEATURE_3D_DIM/2) + z;

			if (iListPos[RandomIndexY[i].x+RandomIndexY[i].y*Feature3D::FEATURE_3D_DIM+RandomIndexY[i].z*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM]>0) {
				continue;
			}
			else{
				iListPos[RandomIndexX[i].x+RandomIndexX[i].y*Feature3D::FEATURE_3D_DIM+RandomIndexX[i].z*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM]+=1;
				iListPos[RandomIndexY[i].x+RandomIndexY[i].y*Feature3D::FEATURE_3D_DIM+RandomIndexY[i].z*Feature3D::FEATURE_3D_DIM*Feature3D::FEATURE_3D_DIM]+=1;
				i++;
			}
		}*/
		int xyz0[3]={5,5,5};
		int xyz1[192]={5,5,4,5,5,6,2,8,5,6,2,4,5,6,9,2,5,5,6,5,8,5,4,1,4,5,9,2,5,3,4,4,5,5,3,2,7,5,3,5,7,4,5,5,2,6,6,2,4,5,4,7,7,6,6,1,5,5,7,3,5,5,3,4,5,7,6,4,8,8,8,4,6,4,7,4,7,5,5,6,3,5,7,5,4,3,7,4,7,2,5,4,2,5,6,5,5,5,1,5,4,6,6,5,4,3,5,6,6,5,7,2,4,5,5,4,3,7,3,4,5,5,9,1,5,4,8,5,7,2,5,2,5,5,7,4,5,2,5,7,8,3,3,2,4,6,5,5,3,5,7,6,5,5,4,7,6,3,5,5,5,8,9,4,5,7,5,5,6,7,3,4,5,5,3,5,8,6,5,3,6,1,3,3,4,3,5,6,4,3,4,5};
		for(int i=0; i<64; i++){
			RandomIndexX[i].x = xyz0[0];
			RandomIndexX[i].y = xyz0[1];
			RandomIndexX[i].z = xyz0[2];
			RandomIndexY[i].x = xyz1[i*3];
			RandomIndexY[i].y = xyz1[i*3+1];
			RandomIndexY[i].z = xyz1[i*3+2];
		}
	}
	return 0;
}

//
// sphericalToCartesian()
//
// Convert polar grid data to cartesian coordinates
// 0 < Zenith Angle < pi
// 0 < Azimuth Angles < 2pi
//
int sphericalToCartesian(float fmagn, float fZenithAng, float fAzimuthAng, int &x, int &y, int &z){
	x=fmagn*sin(fZenithAng)*cos(fAzimuthAng);
	y=fmagn*sin(fZenithAng)*sin(fAzimuthAng);
	z=fmagn*cos(fZenithAng);
	return 0;
}

//
// msResampleFeaturesBRIEF()
//
// Resample as BRIEF Descriptor (Binary Robust Independent Elementary Features)
//
//
int
msResampleFeaturesBRIEF(
	Feature3D &feat3D,
	vector<LOCATION_VALUE_XYZ> &RandomIndexX,
	vector<LOCATION_VALUE_XYZ> &RandomIndexY
)
{

	FEATUREIO fioImg;
	fioImg.x = fioImg.z = fioImg.y = Feature3D::FEATURE_3D_DIM;
	fioImg.pfVectors = &(feat3D.data_zyx[0][0][0]);
	fioImg.pfMeans = 0;
	fioImg.pfVarrs = 0;
	fioImg.iFeaturesPerVector = 1;
	fioImg.t = 1;

	Feature3D fT0;

	FEATUREIO fioImg2;
	fioImg2.x = fioImg2.z = fioImg2.y = Feature3D::FEATURE_3D_DIM;
	fioImg2.pfVectors = &(fT0.data_zyx[0][0][0]);
	fioImg2.pfMeans = 0;
	fioImg2.pfVarrs = 0;
	fioImg2.iFeaturesPerVector = 1;
	fioImg2.t = 1;

	Feature3D fT1;

	FEATUREIO fioImg3;
	fioImg3.x = fioImg3.z = fioImg3.y = Feature3D::FEATURE_3D_DIM;
	fioImg3.pfVectors = &(fT1.data_zyx[0][0][0]);
	fioImg3.pfMeans = 0;
	fioImg3.pfVarrs = 0;
	fioImg3.iFeaturesPerVector = 1;
	fioImg3.t = 1;

	float fSigmaExtra = sqrt(1.6*1.6 - 0.5*0.5);
	assert(RandomIndexX.size()==RandomIndexY.size());
	float descriptor;
	if (RandomIndexX.size()<Feature3DInfo::FEATURE_3D_PCS) {
		assert(0);
		return 0;
	}
	gb3d_blur3d(fioImg, fioImg3, fioImg2, 0.95, 0.01, 0);
	for (int i = 0; i < Feature3DInfo::FEATURE_3D_PCS; i++) {
		//printf("COMPARING %d %d %d with %d %d %d \n", RandomIndexX[i].x, RandomIndexX[i].y, RandomIndexX[i].z, RandomIndexY[i].x, RandomIndexY[i].y, RandomIndexY[i].z);
		descriptor=fioImg2.pfVectors[RandomIndexX[i].x+RandomIndexX[i].y*fioImg.x+RandomIndexX[i].z*fioImg.x*fioImg.y]-fioImg2.pfVectors[RandomIndexY[i].x+RandomIndexY[i].y*fioImg.x+RandomIndexY[i].z*fioImg.x*fioImg.y];

		// BRIEF METHOD
		//feat3D.m_pfPC[i]=descriptor<0;

		// RRIEF Method
		feat3D.m_pfPC[i]=descriptor;

		// NRRIEF Method
		//int eucDist = euclidean_distance_3d(RandomIndexX[i].x, RandomIndexY[i].x, RandomIndexX[i].y, RandomIndexY[i].y,RandomIndexX[i].z, RandomIndexY[i].z);
		//feat3D.m_pfPC[i]=descriptor/eucDist;
	}

	return 0;
}

int euclidean_distance_3d(int x1, int x2, int y1, int y2, int z1, int z2){
	float fdx = x1 - x2;
	float fdy = y1 - y2;
	float fdz = z1 - z2;
	return sqrt(fdx*fdx + fdy*fdy + fdz*fdz);
}

int hammingDistance(int n1, int n2)
{
    int x = n1 ^ n2;
    int setBits = 0;

    while (x > 0) {
        setBits += x & 1;
        x >>= 1;
    }

    return setBits;
}

#define ORIBINS 32


void
mult_4x4_vector(
	float *mat,
	float *vec_in,
	float *vec_out
)
{
	for (int i = 0; i < 4; i++)
	{
		vec_out[i] = 0;
		for (int j = 0; j < 4; j++)
		{
			vec_out[i] += mat[i * 4 + j] * vec_in[j];
		}
	}
}


void
vec3D_norm_3d(
	float *pf1
)
{
	float fSumSqr = pf1[0] * pf1[0] + pf1[1] * pf1[1] + pf1[2] * pf1[2];
	if (fSumSqr > 0)
	{
		float fDiv = 1.0 / sqrt(fSumSqr);
		pf1[0] *= fDiv;
		pf1[1] *= fDiv;
		pf1[2] *= fDiv;
	}
	else
	{
		pf1[0] = 1;
		pf1[1] = 0;
		pf1[2] = 0;
	}
}

float
vec3D_mag(
	float *pf1
)
{
	float fSumSqr = pf1[0] * pf1[0] + pf1[1] * pf1[1] + pf1[2] * pf1[2];
	if (fSumSqr > 0)
	{
		return sqrt(fSumSqr);
	}
	else
	{
		return 0;
	}
}


//
// validateDifferencePeak3D()
//
// Returns 1 if lvPeak is a peak compared to fio-fio2, 0 otherwise.
//
static int
validateDifferencePeak3D(
	LOCATION_VALUE_XYZ &lvPeak,
	FEATUREIO &fio,
	FEATUREIO &fio2
)
{
	// Set up neighbourhood
	int iNeighbourIndexCount;
	int piNeighbourIndices[27];

	int iZStart, iZCount;

	if (fio.z > 1)
	{
		// 3D
		iNeighbourIndexCount = 26;
		iZStart = 1;
		iZCount = fio.z - 1;

		piNeighbourIndices[0] = -fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[1] = -fio.x*fio.y - fio.x;
		piNeighbourIndices[2] = -fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[3] = -fio.x*fio.y - 1;
		piNeighbourIndices[4] = -fio.x*fio.y;
		piNeighbourIndices[5] = -fio.x*fio.y + 1;
		piNeighbourIndices[6] = -fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[7] = -fio.x*fio.y + fio.x;
		piNeighbourIndices[8] = -fio.x*fio.y + fio.x + 1;

		piNeighbourIndices[9] = -fio.x - 1;
		piNeighbourIndices[10] = -fio.x;
		piNeighbourIndices[11] = -fio.x + 1;
		piNeighbourIndices[12] = -1;
		piNeighbourIndices[13] = 1;
		piNeighbourIndices[14] = fio.x - 1;
		piNeighbourIndices[15] = fio.x;
		piNeighbourIndices[16] = fio.x + 1;

		piNeighbourIndices[17] = fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[18] = fio.x*fio.y - fio.x;
		piNeighbourIndices[19] = fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[20] = fio.x*fio.y - 1;
		piNeighbourIndices[21] = fio.x*fio.y;
		piNeighbourIndices[22] = fio.x*fio.y + 1;
		piNeighbourIndices[23] = fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[24] = fio.x*fio.y + fio.x;
		piNeighbourIndices[25] = fio.x*fio.y + fio.x + 1;
	}
	else
	{
		// 2D
		iNeighbourIndexCount = 8;
		iZStart = 0;
		iZCount = 1;

		piNeighbourIndices[0] = -fio.x - 1;
		piNeighbourIndices[1] = -fio.x;
		piNeighbourIndices[2] = -fio.x + 1;
		piNeighbourIndices[3] = -1;
		piNeighbourIndices[4] = 1;
		piNeighbourIndices[5] = fio.x - 1;
		piNeighbourIndices[6] = fio.x;
		piNeighbourIndices[7] = fio.x + 1;
	}

	int iZIndex = lvPeak.z*fio.x*fio.y;
	int iYIndex = iZIndex + lvPeak.y*fio.x;
	int iIndex = iYIndex + lvPeak.x;

	int bPeak = 1;
	float fCenterValue = lvPeak.fValue;

	// Compute DoG value for center
	float fNeighValue = fio.pfVectors[iIndex*fio.iFeaturesPerVector]
		- fio2.pfVectors[iIndex*fio.iFeaturesPerVector];
	bPeak &= (fNeighValue < fCenterValue);

	for (int n = 0; n < iNeighbourIndexCount && bPeak; n++)
	{
		int iNeighbourIndex = iIndex + piNeighbourIndices[n];
		fNeighValue = fio.pfVectors[iNeighbourIndex*fio.iFeaturesPerVector]
			- fio2.pfVectors[iNeighbourIndex*fio.iFeaturesPerVector];

		// A peak means all neighbours are lower than the center value
		bPeak &= (fNeighValue < fCenterValue);
	}
	return bPeak;
}

//
// validateDifferenceValley3D()
//
// Returns 1 if lvPeak is a valley compared to fio1-fio2, 0 otherwise.
//
static int
validateDifferenceValley3D(
	LOCATION_VALUE_XYZ &lvPeak,
	FEATUREIO &fio,
	FEATUREIO &fio2
)
{
	// Set up neighbourhood
	int iNeighbourIndexCount;
	int piNeighbourIndices[27];

	int iZStart, iZCount;

	if (fio.z > 1)
	{
		// 3D
		iNeighbourIndexCount = 26;
		iZStart = 1;
		iZCount = fio.z - 1;

		piNeighbourIndices[0] = -fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[1] = -fio.x*fio.y - fio.x;
		piNeighbourIndices[2] = -fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[3] = -fio.x*fio.y - 1;
		piNeighbourIndices[4] = -fio.x*fio.y;
		piNeighbourIndices[5] = -fio.x*fio.y + 1;
		piNeighbourIndices[6] = -fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[7] = -fio.x*fio.y + fio.x;
		piNeighbourIndices[8] = -fio.x*fio.y + fio.x + 1;

		piNeighbourIndices[9] = -fio.x - 1;
		piNeighbourIndices[10] = -fio.x;
		piNeighbourIndices[11] = -fio.x + 1;
		piNeighbourIndices[12] = -1;
		piNeighbourIndices[13] = 1;
		piNeighbourIndices[14] = fio.x - 1;
		piNeighbourIndices[15] = fio.x;
		piNeighbourIndices[16] = fio.x + 1;

		piNeighbourIndices[17] = fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[18] = fio.x*fio.y - fio.x;
		piNeighbourIndices[19] = fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[20] = fio.x*fio.y - 1;
		piNeighbourIndices[21] = fio.x*fio.y;
		piNeighbourIndices[22] = fio.x*fio.y + 1;
		piNeighbourIndices[23] = fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[24] = fio.x*fio.y + fio.x;
		piNeighbourIndices[25] = fio.x*fio.y + fio.x + 1;
	}
	else
	{
		// 2D
		iNeighbourIndexCount = 8;
		iZStart = 0;
		iZCount = 1;

		piNeighbourIndices[0] = -fio.x - 1;
		piNeighbourIndices[1] = -fio.x;
		piNeighbourIndices[2] = -fio.x + 1;
		piNeighbourIndices[3] = -1;
		piNeighbourIndices[4] = 1;
		piNeighbourIndices[5] = fio.x - 1;
		piNeighbourIndices[6] = fio.x;
		piNeighbourIndices[7] = fio.x + 1;
	}

	int iZIndex = lvPeak.z*fio.x*fio.y;
	int iYIndex = iZIndex + lvPeak.y*fio.x;
	int iIndex = iYIndex + lvPeak.x;

	int bPeak = 1;
	float fCenterValue = lvPeak.fValue;

	// Compute DoG value for center
	float fNeighValue = fio.pfVectors[iIndex*fio.iFeaturesPerVector]
		- fio2.pfVectors[iIndex*fio.iFeaturesPerVector];
	bPeak &= (fNeighValue > fCenterValue);

	for (int n = 0; n < iNeighbourIndexCount && bPeak; n++)
	{
		int iNeighbourIndex = iIndex + piNeighbourIndices[n];
		fNeighValue = fio.pfVectors[iNeighbourIndex*fio.iFeaturesPerVector]
			- fio2.pfVectors[iNeighbourIndex*fio.iFeaturesPerVector];

		// A peak means all neighbours are lower than the center value
		bPeak &= (fNeighValue > fCenterValue);
	}
	return bPeak;
}


//
// generateFeatures3D_efficient()
//
//
//
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
	float fEigThres
)
{
	Feature3D feat3D;
	FEATUREIO featSample;
	featSample.device=-1;
	featSample.x = featSample.y = featSample.z = Feature3D::FEATURE_3D_DIM;
	featSample.t = 1;
	featSample.iFeaturesPerVector = 1;
	fioAllocate(featSample);

	FEATUREIO featJunk; std::memset(&featJunk, 0, sizeof(featJunk));

	int bInterpolate = 1;

	for (int i = 0; i < lvaMinima.iCount; i++)
	{
		if (!bInterpolate)
		{
			feat3D.x = lvaMinima.plvz[i].x;
			feat3D.y = lvaMinima.plvz[i].y;
			feat3D.z = lvaMinima.plvz[i].z;
			feat3D.scale = 2 * fScale;
		}
		else
		{
			interpolate_discrete_3D_point(
				fioC,
				lvaMinima.plvz[i].x, lvaMinima.plvz[i].y, lvaMinima.plvz[i].z,
				feat3D.x, feat3D.y, feat3D.z);
			feat3D.scale = 2 * interpolate_extremum_quadratic(
				fScaleH, fScaleC, fScaleL,
				vecMinH[i],
				fioGetPixel(fioC, lvaMinima.plvz[i].x, lvaMinima.plvz[i].y, lvaMinima.plvz[i].z),
				vecMinL[i]
			);
		}
		// Convert pixel locations to subpixel precision
		feat3D.x += 0.5f;
		feat3D.y += 0.5f;
		feat3D.z += 0.5f;

		feat3D.m_uiInfo &= ~INFO_FLAG_MIN0MAX1;
		generateFeature3D(feat3D, featSample, fioImg, featJunk, featJunk, featJunk, vecFeats, fEigThres);
	}
	for (int i = 0; i < lvaMaxima.iCount; i++)
	{

		if (!bInterpolate)
		{
			feat3D.x = lvaMaxima.plvz[i].x;
			feat3D.y = lvaMaxima.plvz[i].y;
			feat3D.z = lvaMaxima.plvz[i].z;
			feat3D.scale = 2 * fScale;
		}
		else
		{
			interpolate_discrete_3D_point(
				fioC,
				lvaMaxima.plvz[i].x, lvaMaxima.plvz[i].y, lvaMaxima.plvz[i].z,
				feat3D.x, feat3D.y, feat3D.z);
			feat3D.scale = 2 * interpolate_extremum_quadratic(
				fScaleH, fScaleC, fScaleL,
				vecMaxH[i],
				fioGetPixel(fioC, lvaMaxima.plvz[i].x, lvaMaxima.plvz[i].y, lvaMaxima.plvz[i].z),
				vecMaxL[i]
			);
		}
		// Convert pixel locations to subpixel precision
		feat3D.x += 0.5f;
		feat3D.y += 0.5f;
		feat3D.z += 0.5f;

		feat3D.m_uiInfo |= INFO_FLAG_MIN0MAX1;
		generateFeature3D(feat3D, featSample, fioImg, featJunk, featJunk, featJunk, vecFeats, fEigThres);
	}
	fioDelete(featSample);
	return 0;
}

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
)
{
	Feature3D feat3D;
	FEATUREIO featSample;
	featSample.x = featSample.y = featSample.z = Feature3D::FEATURE_3D_DIM;
	featSample.t = 1;
	featSample.iFeaturesPerVector = 1;
	fioAllocate(featSample);
	for (int i = 0; i < lvaMinima.iCount; i++)
	{
		feat3D.x = lvaMinima.plvz[i].x;
		feat3D.y = lvaMinima.plvz[i].y;
		feat3D.z = lvaMinima.plvz[i].z;
		feat3D.scale = 2 * interpolate_extremum_quadratic(
			fScaleH, fScaleC, fScaleL,
			fioGetPixel(fioH, lvaMinima.plvz[i].x, lvaMinima.plvz[i].y, lvaMinima.plvz[i].z),
			fioGetPixel(fioC, lvaMinima.plvz[i].x, lvaMinima.plvz[i].y, lvaMinima.plvz[i].z),
			fioGetPixel(fioL, lvaMinima.plvz[i].x, lvaMinima.plvz[i].y, lvaMinima.plvz[i].z)
		);
		feat3D.eigs[0] = feat3D.eigs[1] = feat3D.eigs[2] = 1;

		if (lvaMaxima.plvz[i].fValue < 0.0001f)
		{
			continue;
		}

		// Save joint histogram data for future prior
		for (int iFeat = 0; iFeat < iImages; iFeat++)
		{
			float *pfProbVec = fioGetVector(fioImages[iFeat], lvaMinima.plvz[i].x, lvaMinima.plvz[i].y, lvaMinima.plvz[i].z);
			feat3D.data_zyx[0][0][iFeat] = *pfProbVec;
		}

		{
			feat3D.m_uiInfo &= 0xFFFFFFEF;
			vecFeats.push_back(feat3D);
		}
	}
	for (int i = 0; i < lvaMaxima.iCount; i++)
	{
		feat3D.x = lvaMaxima.plvz[i].x;
		feat3D.y = lvaMaxima.plvz[i].y;
		feat3D.z = lvaMaxima.plvz[i].z;
		feat3D.scale = 2 * interpolate_extremum_quadratic(
			fScaleH, fScaleC, fScaleL,
			fioGetPixel(fioH, lvaMaxima.plvz[i].x, lvaMaxima.plvz[i].y, lvaMaxima.plvz[i].z),
			fioGetPixel(fioC, lvaMaxima.plvz[i].x, lvaMaxima.plvz[i].y, lvaMaxima.plvz[i].z),
			fioGetPixel(fioL, lvaMaxima.plvz[i].x, lvaMaxima.plvz[i].y, lvaMaxima.plvz[i].z)
		);
		feat3D.eigs[0] = feat3D.eigs[1] = feat3D.eigs[2] = 1;

		if (lvaMaxima.plvz[i].fValue < 0.0001f)
		{
			continue;
		}

		// Save joint histogram data for future prior
		for (int iFeat = 0; iFeat < iImages; iFeat++)
		{
			float *pfProbVec = fioGetVector(fioImages[iFeat], lvaMaxima.plvz[i].x, lvaMaxima.plvz[i].y, lvaMaxima.plvz[i].z);
			feat3D.data_zyx[0][0][iFeat] = *pfProbVec;
		}

		{
			feat3D.m_uiInfo |= 0x00000010;
			vecFeats.push_back(feat3D);
		}
	}
	fioDelete(featSample);
	return 0;
}

int
detectExtrema4D_test_interleave(
	FEATUREIO fioD0, FEATUREIO fioD1, FEATUREIO fioSumOfSign,
	LOCATION_VALUE_XYZ_ARRAY &lvaMinima,
	LOCATION_VALUE_XYZ_ARRAY &lvaMaxima,
	int best_device_id
){
	if (best_device_id!=-1) {
		detectExtrema4D_test_cuda(
			fioD0, fioD1, fioSumOfSign,
			lvaMinima,
			lvaMaxima,
			best_device_id
		);
	}
	else{
		detectExtrema4D_test(
			&fioD0, &fioD1, 0,
			lvaMinima,
			lvaMaxima
		);
	}
	return 1;
}

int
detectExtrema4D_test(
	FEATUREIO *pfioH,	// Hi res image
	FEATUREIO *pfioC,	// Center image
	FEATUREIO *pfioL,	// Lo res image
	LOCATION_VALUE_XYZ_ARRAY	&lvaMinima,
	LOCATION_VALUE_XYZ_ARRAY	&lvaMaxima,
	int bDense
)
{
	// Perform only 3D
	EXTREMA_STRUCT es;
	es.pfioH = pfioH;
	es.pfioL = pfioL;
	es.bDense = bDense;

	//regFindFEATUREIOValleys(lvaMinima, *pfioC, valleyFunction4D, &es);
	//regFindFEATUREIOPeaks(lvaMaxima, *pfioC, peakFunction4D, &es);
	regFindFEATUREIO_interleave(lvaMaxima, lvaMinima, *pfioC, peakFunction4D, valleyFunction4D, &es);

	return 1;
}

float
vec3D_dot_3d(
	const float *pv1,
	const float *pv2
)
{
	return pv1[0] * pv2[0] + pv1[1] * pv2[1] + pv1[2] * pv2[2];
}

void
msNormalizeDataPositive(
	float *pfVec,
	int iLength
)
{
	float fMin = 100000;

	for (int zz = 0; zz < iLength; zz++)
	{
		if (pfVec[zz] < fMin)
		{
			fMin = pfVec[zz];
		}

	}
	float fSumSqr = 0;

	for (int zz = 0; zz < iLength; zz++)
	{
		pfVec[zz] -= fMin;
		fSumSqr += pfVec[zz] * pfVec[zz];
	}
	float fDiv = 1.0f / (float)sqrt(fSumSqr);
	fSumSqr = 0;

	for (int zz = 0; zz < iLength; zz++)
	{
		pfVec[zz] *= fDiv;
		fSumSqr += pfVec[zz] * pfVec[zz];
	}
}


void
interpolate_discrete_3D_point(
	FEATUREIO &fioC,	// Center image: for interpolating feature geometry
	int ix, int iy, int iz,
	float &fx, float &fy, float &fz
)
{
	fx = interpolate_extremum_quadratic(
		ix - 1, ix, ix + 1,
		fioGetPixel(fioC, ix - 1, iy, iz),
		fioGetPixel(fioC, ix, iy, iz),
		fioGetPixel(fioC, ix + 1, iy, iz)
	);
	fy = interpolate_extremum_quadratic(
		iy - 1, iy, iy + 1,
		fioGetPixel(fioC, ix, iy - 1, iz),
		fioGetPixel(fioC, ix, iy, iz),
		fioGetPixel(fioC, ix, iy + 1, iz)
	);
	fz = interpolate_extremum_quadratic(
		iz - 1, iz, iz + 1,
		fioGetPixel(fioC, ix, iy, iz - 1),
		fioGetPixel(fioC, ix, iy, iz),
		fioGetPixel(fioC, ix, iy, iz + 1)
	);
}

double
interpolate_extremum_quadratic(
	double x0, // Three coordinates x
	double x1,
	double x2,
	double fx0, // Three functions of coordinates f(x)
	double fx1,
	double fx2
)
{
	// fx1 must be either a minimum or a maximum
	if (!(fx1 < fx0 && fx1 < fx2) && !(fx1 > fx0 && fx1 > fx2))
	{
		printf("%f\t%f\t%f\n", fx0, fx1, fx2);
		assert(0);
		return x1;
	}

	double a1 = x0*x0, b1 = x0, c1 = 1;
	double a2 = x1*x1, b2 = x1, c2 = 1;
	double a3 = x2*x2, b3 = x2, c3 = 1;
	double d1 = fx0, d2 = fx1, d3 = fx2;
	double det, detx, dety, detz;

	det = finddet(a1, a2, a3, b1, b2, b3, c1, c2, c3);   /* Find determinants */
	detx = finddet(d1, d2, d3, b1, b2, b3, c1, c2, c3);
	dety = finddet(a1, a2, a3, d1, d2, d3, c1, c2, c3);
	detz = finddet(a1, a2, a3, b1, b2, b3, d1, d2, d3);

	if (d1 == 0 && d2 == 0 && d3 == 0 && det == 0)
	{
		printf("\n Infinite Solutions\n ");
	}
	else if (d1 == 0 && d2 == 0 && d3 == 0 && det != 0)
	{
		printf("\n x=0\n y=0, \n z=0\n ");
	}
	else if (det != 0)
	{
		if (detx != 0)
		{
			// Valid interpolated value
			return dety / (-2.0*detx);
		}
	}
	else if (det == 0 && detx == 0 && dety == 0 && detz == 0)
	{
		printf("\n Infinite Solutions\n ");
	}
	else
	{
		printf("No Solution\n ");
	}

	// Return x1 (uninterpolated peak) by default
	return x1;
}

//
// generateFeature3D()
//
// Determine orientation parameters for feat3D, based on ocation & scale
// of feat3D.
//
int
generateFeature3D(
	Feature3D &feat3D, // Feature geometrical information
	FEATUREIO &fioSample, // Sub-image sample
	FEATUREIO &fioImg, // Original image
	FEATUREIO &fioDx, // Derivative images
	FEATUREIO &fioDy,
	FEATUREIO &fioDz,
	vector<Feature3D> &vecFeats,
	float fEigThres,// = -1,// Threshold on eigenvalues, discard if
	int bReorientedFeatures// = 1 Wether to reorient features or not, default yes (1)
)
{
	// Determine feature orientation - to be scale-consistent,
	// derivatives should be calculated on rescaled image ...
	//   ..and never again will I calculate derivatives on a non-scaled image ...

	// Determine feature image content - init rotation to identity
	std::memset(&(feat3D.ori[0][0]), 0, sizeof(feat3D.ori));
	feat3D.ori[0][0] = 1; feat3D.ori[1][1] = 1; feat3D.ori[2][2] = 1;
	if (sampleImage3D(feat3D, fioSample, fioImg, fioDx, fioDy, fioDz) != 0)
	{
		return -1;
	}
	float fMaxPixel = fioGetPixel(fioSample, 0, 0, 0);
	float fMinPixel = fioGetPixel(fioSample, 0, 0, 0);
	for (int zz = 0; zz < Feature3D::FEATURE_3D_DIM; zz++)
	{
		for (int yy = 0; yy < Feature3D::FEATURE_3D_DIM; yy++)
		{
			for (int xx = 0; xx < Feature3D::FEATURE_3D_DIM; xx++)
			{
				feat3D.data_zyx[zz][yy][xx] = fioGetPixel(fioSample, xx, yy, zz);
			}
		}
	}

	feat3D.NormalizeData();
	//// Determine feature eigenorientation
	if (determineOrientation3D(feat3D))
	{
		return -1;
	}
	float fEigSum = feat3D.eigs[0] + feat3D.eigs[1] + feat3D.eigs[2];
	float fEigPrd = feat3D.eigs[0] * feat3D.eigs[1] * feat3D.eigs[2];
	float fEigSumProd = fEigSum*fEigSum*fEigSum;

	static float fMaxRatio = -1;
	static float fMinRatio = 2;
	float fRatio = (27.0f*fEigPrd) / fEigSumProd;
	if (fRatio > fMaxRatio)
	{
		fMaxRatio = fRatio;
	}
	if (fRatio < fMinRatio)
	{
		fMinRatio = fRatio;
	}
	if (fEigSumProd < fEigThres*fEigPrd || fEigThres < 0)
	{
	}
	else
	{
		return -1;
	}

	// Flag as not reoriented
	feat3D.m_uiInfo &= ~INFO_FLAG_REORIENT;
	vecFeats.push_back(feat3D);

	if (!bReorientedFeatures)
	{
		return 0;
	}

#ifdef INCLUDE_EIGENORIENTATION_FEATURE
	// Sample feature eigenorientation
	if (sampleImage3D(feat3D, fioSample, fioImg, fioDx, fioDy, fioDz) != 0)
	{
		return -1;
	}
	for (int zz = 0; zz < Feature3D::FEATURE_3D_DIM; zz++)
	{
		for (int yy = 0; yy < Feature3D::FEATURE_3D_DIM; yy++)
		{
			for (int xx = 0; xx < Feature3D::FEATURE_3D_DIM; xx++)
			{
				feat3D.data_zyx[zz][yy][xx] = fioGetPixel(fioSample, xx, yy, zz);
			}
		}
	}

	vecFeats.push_back(feat3D);

	// *******************************
	// // Determine feature image content - init rotation to identity

	memset(&(feat3D.ori[0][0]), 0, sizeof(feat3D.ori));
	feat3D.ori[0][0] = 1; feat3D.ori[1][1] = 1; feat3D.ori[2][2] = 1;
	if (sampleImage3D(feat3D, fioSample, fioImg, fioDx, fioDy, fioDz) != 0)
	{
		return -1;
	}
	// Copy sampled data
	for (int zz = 0; zz < Feature3D::FEATURE_3D_DIM; zz++)
	{
		for (int yy = 0; yy < Feature3D::FEATURE_3D_DIM; yy++)
		{
			for (int xx = 0; xx < Feature3D::FEATURE_3D_DIM; xx++)
			{
				feat3D.data_zyx[zz][yy][xx] = fioGetPixel(fioSample, xx, yy, zz);
			}
		}
	}

#endif
	// There are multiple orientations, create feature for each
	// Pass in enough room for 30 rotation matrices (each consists of three 3D unit vectors)
	float pfOri[30 * 9];
	float iMaxOri = 30;
	int iOrientationsFound = determineCanonicalOrientation3D(feat3D, pfOri, iMaxOri);
	for (int iOri = 0; iOri < iOrientationsFound; iOri++)
	{
		// Copy orientation vector
		float *pfOriStart = pfOri + 9 * iOri;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				feat3D.ori[i][j] = pfOriStart[i * 3 + j];
			}
		}

		// Resample image with new orientation
		if (sampleImage3D(feat3D, fioSample, fioImg, fioDx, fioDy, fioDz) != 0)
		{
			continue;
		}

		// Copy sampled data to feature
		for (int zz = 0; zz < Feature3D::FEATURE_3D_DIM; zz++)
		{
			for (int yy = 0; yy < Feature3D::FEATURE_3D_DIM; yy++)
			{
				for (int xx = 0; xx < Feature3D::FEATURE_3D_DIM; xx++)
				{
					feat3D.data_zyx[zz][yy][xx] = fioGetPixel(fioSample, xx, yy, zz);
				}
			}
		}

		// Flag as reoriented
		feat3D.m_uiInfo |= INFO_FLAG_REORIENT;
		vecFeats.push_back(feat3D);
	}

	return 0;
}

static int
regFindFEATUREIOValleys(
	LOCATION_VALUE_XYZ_ARRAY &lvaPeaks,
	FEATUREIO &fio,
	EXTREMA_FUNCTION pCallback,
	EXTREMA_STRUCT *pES
)
{
	// Set up neighbourhood
	int iNeighbourIndexCount;
	int piNeighbourIndices[27];

	int iZStart, iZCount;

	if (fio.z > 1)
	{
		// 3D
		iNeighbourIndexCount = 26;
		iZStart = 1;
		iZCount = fio.z - 1;

		piNeighbourIndices[0] = -fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[1] = -fio.x*fio.y - fio.x;
		piNeighbourIndices[2] = -fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[3] = -fio.x*fio.y - 1;
		piNeighbourIndices[4] = -fio.x*fio.y;
		piNeighbourIndices[5] = -fio.x*fio.y + 1;
		piNeighbourIndices[6] = -fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[7] = -fio.x*fio.y + fio.x;
		piNeighbourIndices[8] = -fio.x*fio.y + fio.x + 1;

		piNeighbourIndices[9] = -fio.x - 1;
		piNeighbourIndices[10] = -fio.x;
		piNeighbourIndices[11] = -fio.x + 1;
		piNeighbourIndices[12] = -1;
		piNeighbourIndices[13] = 1;
		piNeighbourIndices[14] = fio.x - 1;
		piNeighbourIndices[15] = fio.x;
		piNeighbourIndices[16] = fio.x + 1;

		piNeighbourIndices[17] = fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[18] = fio.x*fio.y - fio.x;
		piNeighbourIndices[19] = fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[20] = fio.x*fio.y - 1;
		piNeighbourIndices[21] = fio.x*fio.y;
		piNeighbourIndices[22] = fio.x*fio.y + 1;
		piNeighbourIndices[23] = fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[24] = fio.x*fio.y + fio.x;
		piNeighbourIndices[25] = fio.x*fio.y + fio.x + 1;
	}
	else
	{
		// 2D
		iNeighbourIndexCount = 8;
		iZStart = 0;
		iZCount = 1;

		piNeighbourIndices[0] = -fio.x - 1;
		piNeighbourIndices[1] = -fio.x;
		piNeighbourIndices[2] = -fio.x + 1;
		piNeighbourIndices[3] = -1;
		piNeighbourIndices[4] = 1;
		piNeighbourIndices[5] = fio.x - 1;
		piNeighbourIndices[6] = fio.x;
		piNeighbourIndices[7] = fio.x + 1;
	}

	lvaPeaks.iCount = 0;

	for (int z = iZStart; z < iZCount; z++)
	{
		int iZIndex = z*fio.x*fio.y;
		for (int y = 1; y < fio.y - 1; y++)
		{
			int iYIndex = iZIndex + y*fio.x;
			for (int x = 1; x < fio.x - 1; x++)
			{
				int iIndex = iYIndex + x;

				int bPeak = 1;
				float fCenterValue = fio.pfVectors[iIndex*fio.iFeaturesPerVector];

				for (int n = 0; n < iNeighbourIndexCount && bPeak; n++)
				{
					int iNeighbourIndex = iIndex + piNeighbourIndices[n];
					float fNeighValue = fio.pfVectors[iNeighbourIndex*fio.iFeaturesPerVector];

					// A valley means all neighbours are higher than the center value
					bPeak &= (fNeighValue > fCenterValue);
				}

				if (bPeak)
				{
					if (pCallback)
					{
						pES->iCurrIndex = iIndex;
						pES->iNeighbourCount = iNeighbourIndexCount;
						pES->lva.x = x;
						pES->lva.y = y;
						pES->lva.z = z;
						pES->lva.fValue = fCenterValue;
						pES->piNeighbourIndices = piNeighbourIndices;

						if (pCallback(pES))
						{
							if (lvaPeaks.plvz)
							{
								lvaPeaks.plvz[lvaPeaks.iCount].x = x;
								lvaPeaks.plvz[lvaPeaks.iCount].y = y;
								lvaPeaks.plvz[lvaPeaks.iCount].z = z;
								lvaPeaks.plvz[lvaPeaks.iCount].fValue = fCenterValue;
							}
							lvaPeaks.iCount++;
						}
					}
				}
			}
		}
	}

	return 1;
}

static int
regFindFEATUREIOPeaks(
	LOCATION_VALUE_XYZ_ARRAY &lvaPeaks,
	FEATUREIO &fio,
	EXTREMA_FUNCTION pCallback,
	EXTREMA_STRUCT *pES
)
{
	// Set up neighbourhood

	int iNeighbourIndexCount;
	int piNeighbourIndices[27];

	int iZStart, iZCount;

	if (fio.z > 1)
	{
		// 3D
		iNeighbourIndexCount = 26;
		iZStart = 1;
		iZCount = fio.z - 1;

		piNeighbourIndices[0] = -fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[1] = -fio.x*fio.y - fio.x;
		piNeighbourIndices[2] = -fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[3] = -fio.x*fio.y - 1;
		piNeighbourIndices[4] = -fio.x*fio.y;
		piNeighbourIndices[5] = -fio.x*fio.y + 1;
		piNeighbourIndices[6] = -fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[7] = -fio.x*fio.y + fio.x;
		piNeighbourIndices[8] = -fio.x*fio.y + fio.x + 1;

		piNeighbourIndices[9] = -fio.x - 1;
		piNeighbourIndices[10] = -fio.x;
		piNeighbourIndices[11] = -fio.x + 1;
		piNeighbourIndices[12] = -1;
		piNeighbourIndices[13] = 1;
		piNeighbourIndices[14] = fio.x - 1;
		piNeighbourIndices[15] = fio.x;
		piNeighbourIndices[16] = fio.x + 1;

		piNeighbourIndices[17] = fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[18] = fio.x*fio.y - fio.x;
		piNeighbourIndices[19] = fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[20] = fio.x*fio.y - 1;
		piNeighbourIndices[21] = fio.x*fio.y;
		piNeighbourIndices[22] = fio.x*fio.y + 1;
		piNeighbourIndices[23] = fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[24] = fio.x*fio.y + fio.x;
		piNeighbourIndices[25] = fio.x*fio.y + fio.x + 1;
	}
	else
	{
		// 2D
		iNeighbourIndexCount = 8;
		iZStart = 0;
		iZCount = 1;

		piNeighbourIndices[0] = -fio.x - 1;
		piNeighbourIndices[1] = -fio.x;
		piNeighbourIndices[2] = -fio.x + 1;
		piNeighbourIndices[3] = -1;
		piNeighbourIndices[4] = 1;
		piNeighbourIndices[5] = fio.x - 1;
		piNeighbourIndices[6] = fio.x;
		piNeighbourIndices[7] = fio.x + 1;
	}

	lvaPeaks.iCount = 0;

	for (int z = iZStart; z < iZCount; z++)
	{
		int iZIndex = z*fio.x*fio.y;
		for (int y = 1; y < fio.y - 1; y++)
		{
			int iYIndex = iZIndex + y*fio.x;
			for (int x = 1; x < fio.x - 1; x++)
			{
				int iIndex = iYIndex + x;

				int bPeak = 1;
				float fCenterValue = fio.pfVectors[iIndex*fio.iFeaturesPerVector];

				for (int n = 0; n < iNeighbourIndexCount && bPeak; n++)
				{
					int iNeighbourIndex = iIndex + piNeighbourIndices[n];
					float fNeighValue = fio.pfVectors[iNeighbourIndex*fio.iFeaturesPerVector];

					// A peak means all neighbours are lower than the center value
					bPeak &= (fNeighValue < fCenterValue);
				}

				if (bPeak)
				{
					if (pCallback)
					{
						pES->iCurrIndex = iIndex;
						pES->iNeighbourCount = iNeighbourIndexCount;
						pES->lva.x = x;
						pES->lva.y = y;
						pES->lva.z = z;
						pES->lva.fValue = fCenterValue;
						pES->piNeighbourIndices = piNeighbourIndices;

						if (pCallback(pES))
						{
							if (lvaPeaks.plvz)
							{
								lvaPeaks.plvz[lvaPeaks.iCount].x = x;
								lvaPeaks.plvz[lvaPeaks.iCount].y = y;
								lvaPeaks.plvz[lvaPeaks.iCount].z = z;
								lvaPeaks.plvz[lvaPeaks.iCount].fValue = fCenterValue;
							}
							lvaPeaks.iCount++;
						}
					}
					else
					{
						// No callback, just add as a peak
						if (lvaPeaks.plvz)
						{
							lvaPeaks.plvz[lvaPeaks.iCount].x = x;
							lvaPeaks.plvz[lvaPeaks.iCount].y = y;
							lvaPeaks.plvz[lvaPeaks.iCount].z = z;
							lvaPeaks.plvz[lvaPeaks.iCount].fValue = fCenterValue;
						}
						lvaPeaks.iCount++;
					}
				}
			}
		}
	}

	return 1;
}

static int
regFindFEATUREIO_interleave(
	LOCATION_VALUE_XYZ_ARRAY &lvaMaxima,
	LOCATION_VALUE_XYZ_ARRAY &lvaMinima,
	FEATUREIO &fio,
	EXTREMA_FUNCTION peakFunction4D,
	EXTREMA_FUNCTION valleyFunction4D,
	EXTREMA_STRUCT *pES){
		if (!peakFunction4D && !valleyFunction4D) {
			return regFindFEATUREIO_NO_CALLBACK(lvaMaxima, lvaMinima, fio, pES);
		}
		else{
			return regFindFEATUREIO(lvaMaxima, lvaMinima, fio, peakFunction4D, valleyFunction4D, pES);
		}
	}


static int
regFindFEATUREIO_NO_CALLBACK(
	LOCATION_VALUE_XYZ_ARRAY &lvaMaxima,
	LOCATION_VALUE_XYZ_ARRAY &lvaMinima,
	FEATUREIO &fio,
	EXTREMA_STRUCT *pES)
{

	int iNeighbourIndexCount;
	int piNeighbourIndices[27];

	int iZStart, iZCount;

	if (fio.z > 1)
	{
		// 3D
		iNeighbourIndexCount = 26;
		iZStart = 1;
		iZCount = fio.z - 1;

		piNeighbourIndices[0] = -fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[1] = -fio.x*fio.y - fio.x;
		piNeighbourIndices[2] = -fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[3] = -fio.x*fio.y - 1;
		piNeighbourIndices[4] = -fio.x*fio.y;
		piNeighbourIndices[5] = -fio.x*fio.y + 1;
		piNeighbourIndices[6] = -fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[7] = -fio.x*fio.y + fio.x;
		piNeighbourIndices[8] = -fio.x*fio.y + fio.x + 1;

		piNeighbourIndices[9] = -fio.x - 1;
		piNeighbourIndices[10] = -fio.x;
		piNeighbourIndices[11] = -fio.x + 1;
		piNeighbourIndices[12] = -1;
		piNeighbourIndices[13] = 1;
		piNeighbourIndices[14] = fio.x - 1;
		piNeighbourIndices[15] = fio.x;
		piNeighbourIndices[16] = fio.x + 1;

		piNeighbourIndices[17] = fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[18] = fio.x*fio.y - fio.x;
		piNeighbourIndices[19] = fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[20] = fio.x*fio.y - 1;
		piNeighbourIndices[21] = fio.x*fio.y;
		piNeighbourIndices[22] = fio.x*fio.y + 1;
		piNeighbourIndices[23] = fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[24] = fio.x*fio.y + fio.x;
		piNeighbourIndices[25] = fio.x*fio.y + fio.x + 1;
	}
	else
	{
		// 2D
		iNeighbourIndexCount = 8;
		iZStart = 0;
		iZCount = 1;

		piNeighbourIndices[0] = -fio.x - 1;
		piNeighbourIndices[1] = -fio.x;
		piNeighbourIndices[2] = -fio.x + 1;
		piNeighbourIndices[3] = -1;
		piNeighbourIndices[4] = 1;
		piNeighbourIndices[5] = fio.x - 1;
		piNeighbourIndices[6] = fio.x;
		piNeighbourIndices[7] = fio.x + 1;
	}

	lvaMaxima.iCount=0;
	lvaMinima.iCount=0;
	int bPeak;
	int signValue;
	for (int z = iZStart; z < iZCount; z++)
	{
		int iZIndex = z*fio.x*fio.y;
		for (int y = 1; y < fio.y - 1; y++)
		{
			int iYIndex = iZIndex + y*fio.x;
			for (int x = 1; x < fio.x - 1; x++)
			{
				int iIndex = iYIndex + x;
				float fCenterValue = fio.pfVectors[iIndex*fio.iFeaturesPerVector];
				int bPeakinit=sign<float>(fCenterValue - fio.pfVectors[iIndex + piNeighbourIndices[0]*fio.iFeaturesPerVector]);;
				bPeak=bPeakinit;
				int bPeakbin=1;
				for (int n = 1; n < iNeighbourIndexCount && bPeakinit && bPeakbin; n++)
				{
					int iNeighbourIndex = iIndex + piNeighbourIndices[n];
					float fNeighValue = fio.pfVectors[iNeighbourIndex*fio.iFeaturesPerVector];
					signValue=sign<float>(fCenterValue - fNeighValue);
					bPeakbin &= (signValue==bPeakinit);
					// A peak means all neighbours are lower than the center value
					bPeak += signValue;
				}
				if (bPeak==iNeighbourIndexCount) { // Peak
					// No callback, just add as a peak
					if (lvaMaxima.plvz)
					{
						lvaMaxima.plvz[lvaMaxima.iCount].x = x;
						lvaMaxima.plvz[lvaMaxima.iCount].y = y;
						lvaMaxima.plvz[lvaMaxima.iCount].z = z;
						lvaMaxima.plvz[lvaMaxima.iCount].fValue = fCenterValue;
					}
					lvaMaxima.iCount++;
				}
				if (bPeak==-iNeighbourIndexCount){ // Valley
					// No callback, just add as a peak
					if (lvaMinima.plvz)
					{
						lvaMinima.plvz[lvaMinima.iCount].x = x;
						lvaMinima.plvz[lvaMinima.iCount].y = y;
						lvaMinima.plvz[lvaMinima.iCount].z = z;
						lvaMinima.plvz[lvaMinima.iCount].fValue = fCenterValue;
					}
					lvaMinima.iCount++;
				}
			}
		}
	}
	return 1;
}

static int
regFindFEATUREIO(
	LOCATION_VALUE_XYZ_ARRAY &lvaMaxima,
	LOCATION_VALUE_XYZ_ARRAY &lvaMinima,
	FEATUREIO &fio,
	EXTREMA_FUNCTION peakFunction4D,
	EXTREMA_FUNCTION valleyFunction4D,
	EXTREMA_STRUCT *pES)
{

	int iNeighbourIndexCount;
	int piNeighbourIndices[27];

	int iZStart, iZCount;

	if (fio.z > 1)
	{
		// 3D
		iNeighbourIndexCount = 26;
		iZStart = 1;
		iZCount = fio.z - 1;

		piNeighbourIndices[0] = -fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[1] = -fio.x*fio.y - fio.x;
		piNeighbourIndices[2] = -fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[3] = -fio.x*fio.y - 1;
		piNeighbourIndices[4] = -fio.x*fio.y;
		piNeighbourIndices[5] = -fio.x*fio.y + 1;
		piNeighbourIndices[6] = -fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[7] = -fio.x*fio.y + fio.x;
		piNeighbourIndices[8] = -fio.x*fio.y + fio.x + 1;

		piNeighbourIndices[9] = -fio.x - 1;
		piNeighbourIndices[10] = -fio.x;
		piNeighbourIndices[11] = -fio.x + 1;
		piNeighbourIndices[12] = -1;
		piNeighbourIndices[13] = 1;
		piNeighbourIndices[14] = fio.x - 1;
		piNeighbourIndices[15] = fio.x;
		piNeighbourIndices[16] = fio.x + 1;

		piNeighbourIndices[17] = fio.x*fio.y - fio.x - 1;
		piNeighbourIndices[18] = fio.x*fio.y - fio.x;
		piNeighbourIndices[19] = fio.x*fio.y - fio.x + 1;
		piNeighbourIndices[20] = fio.x*fio.y - 1;
		piNeighbourIndices[21] = fio.x*fio.y;
		piNeighbourIndices[22] = fio.x*fio.y + 1;
		piNeighbourIndices[23] = fio.x*fio.y + fio.x - 1;
		piNeighbourIndices[24] = fio.x*fio.y + fio.x;
		piNeighbourIndices[25] = fio.x*fio.y + fio.x + 1;
	}
	else
	{
		// 2D
		iNeighbourIndexCount = 8;
		iZStart = 0;
		iZCount = 1;

		piNeighbourIndices[0] = -fio.x - 1;
		piNeighbourIndices[1] = -fio.x;
		piNeighbourIndices[2] = -fio.x + 1;
		piNeighbourIndices[3] = -1;
		piNeighbourIndices[4] = 1;
		piNeighbourIndices[5] = fio.x - 1;
		piNeighbourIndices[6] = fio.x;
		piNeighbourIndices[7] = fio.x + 1;
	}

	lvaMaxima.iCount=0;
	lvaMinima.iCount=0;
	int bPeak;
	int signValue;
	for (int z = iZStart; z < iZCount; z++)
	{
		int iZIndex = z*fio.x*fio.y;
		for (int y = 1; y < fio.y - 1; y++)
		{
			int iYIndex = iZIndex + y*fio.x;
			for (int x = 1; x < fio.x - 1; x++)
			{
				int iIndex = iYIndex + x;
				float fCenterValue = fio.pfVectors[iIndex*fio.iFeaturesPerVector];
				int bPeakinit=sign<float>(fCenterValue - fio.pfVectors[iIndex + piNeighbourIndices[0]*fio.iFeaturesPerVector]);
				bPeak=bPeakinit;
				int bPeakbin=1;
				for (int n = 1; n < iNeighbourIndexCount && bPeakinit && bPeakbin; n++)
				{
					int iNeighbourIndex = iIndex + piNeighbourIndices[n];
					float fNeighValue = fio.pfVectors[iNeighbourIndex*fio.iFeaturesPerVector];
					signValue=sign<float>(fCenterValue - fNeighValue);
					bPeakbin &= (signValue==bPeakinit);
					// A peak means all neighbours are lower than the center value
					bPeak += signValue;
				}
				if (bPeak==iNeighbourIndexCount) { // Peak
					pES->iCurrIndex = iIndex;
					pES->iNeighbourCount = iNeighbourIndexCount;
					pES->lva.x = x;
					pES->lva.y = y;
					pES->lva.z = z;
					pES->lva.fValue = fCenterValue;
					pES->piNeighbourIndices = piNeighbourIndices;

					if (peakFunction4D(pES))
					{
						if (lvaMaxima.plvz)
						{
							lvaMaxima.plvz[lvaMaxima.iCount].x = x;
							lvaMaxima.plvz[lvaMaxima.iCount].y = y;
							lvaMaxima.plvz[lvaMaxima.iCount].z = z;
							lvaMaxima.plvz[lvaMaxima.iCount].fValue = fCenterValue;
						}
						lvaMaxima.iCount++;
					}
				}
				if (bPeak==-iNeighbourIndexCount){ // Valley
					pES->iCurrIndex = iIndex;
					pES->iNeighbourCount = iNeighbourIndexCount;
					pES->lva.x = x;
					pES->lva.y = y;
					pES->lva.z = z;
					pES->lva.fValue = fCenterValue;
					pES->piNeighbourIndices = piNeighbourIndices;

					if (valleyFunction4D(pES))
					{
						if (lvaMinima.plvz)
						{
							lvaMinima.plvz[lvaMinima.iCount].x = x;
							lvaMinima.plvz[lvaMinima.iCount].y = y;
							lvaMinima.plvz[lvaMinima.iCount].z = z;
							lvaMinima.plvz[lvaMinima.iCount].fValue = fCenterValue;
						}
						lvaMinima.iCount++;
					}
				}
			}
		}
	}
	return 1;
}

//
// valleyFunction()
//
// Checks that a particular point is a valley in scales
// above and below the current scale.
//
int
valleyFunction4D(
	EXTREMA_STRUCT *pES
)
{
	int bValley = 1;

	float fNeighValue;
	int iNeighbourIndex;

	// Check center values above/below
	if (pES->pfioH)
	{
		fNeighValue =
			pES->pfioH->pfVectors[pES->iCurrIndex*pES->pfioH->iFeaturesPerVector];
		bValley &= (fNeighValue > pES->lva.fValue);
	}

	if (pES->pfioL)
	{
		fNeighValue =
			pES->pfioL->pfVectors[pES->iCurrIndex*pES->pfioL->iFeaturesPerVector];
		bValley &= (fNeighValue > pES->lva.fValue);
	}

	if (pES->bDense)
	{
		return bValley;
	}

	if (pES->pfioH)
	{
		// Check 26 neighbours (including center) in scale above
		for (int n = 0; n < pES->iNeighbourCount && bValley; n++)
		{
			iNeighbourIndex = pES->iCurrIndex + pES->piNeighbourIndices[n];
			fNeighValue =
				pES->pfioH->pfVectors[iNeighbourIndex*pES->pfioH->iFeaturesPerVector];
			bValley &= (fNeighValue > pES->lva.fValue);
		}
	}

	if (pES->pfioL)
	{
		// Check 26 neighbours (including center) in scale below
		for (int n = 0; n < pES->iNeighbourCount && bValley; n++)
		{
			iNeighbourIndex = pES->iCurrIndex + pES->piNeighbourIndices[n];
			fNeighValue =
				pES->pfioL->pfVectors[iNeighbourIndex*pES->pfioL->iFeaturesPerVector];
			bValley &= (fNeighValue > pES->lva.fValue);
		}
	}

	return bValley;
}
//
// peakFunction()
//
// Checks that a particular point is a peak in scales
// above and below the current scale.
//
int
peakFunction4D(
	EXTREMA_STRUCT *pES
)
{
	int bPeak = 1;

	float fNeighValue;
	int iNeighbourIndex;

	if (pES->pfioH)
	{
		fNeighValue =
			pES->pfioH->pfVectors[pES->iCurrIndex*pES->pfioH->iFeaturesPerVector];
		bPeak &= (fNeighValue < pES->lva.fValue);
	}

	if (pES->pfioL)
	{
		fNeighValue =
			pES->pfioL->pfVectors[pES->iCurrIndex*pES->pfioL->iFeaturesPerVector];
		bPeak &= (fNeighValue < pES->lva.fValue);
	}

	if (pES->bDense)
	{
		return bPeak;
	}

	if (pES->pfioH)
	{
		// Check 26 neighbours in scale above
		for (int n = 0; n < pES->iNeighbourCount && bPeak; n++)
		{
			iNeighbourIndex = pES->iCurrIndex + pES->piNeighbourIndices[n];
			fNeighValue =
				pES->pfioH->pfVectors[iNeighbourIndex*pES->pfioH->iFeaturesPerVector];
			bPeak &= (fNeighValue < pES->lva.fValue);
		}
	}

	if (pES->pfioL)
	{
		// Check 26 neighbours in scale below
		for (int n = 0; n < pES->iNeighbourCount && bPeak; n++)
		{
			iNeighbourIndex = pES->iCurrIndex + pES->piNeighbourIndices[n];
			fNeighValue =
				pES->pfioL->pfVectors[iNeighbourIndex*pES->pfioL->iFeaturesPerVector];
			bPeak &= (fNeighValue < pES->lva.fValue);
		}
	}

	return bPeak;
}

//
// interpolate_peak_quadratic()
//
// Interpolate an extremum (peak or valley) in a function f(x). f(x1) is the uninterpolated extremum.
//
double finddet(double a1, double a2, double a3, double b1, double b2, double b3, double c1, double c2, double c3)
{
	return ((a1*b2*c3) - (a1*b3*c2) - (a2*b1*c3) + (a3*b1*c2) + (a2*b3*c1) - (a3*b2*c1)); /*expansion of a 3x3 determinant*/
}

//
// determineOrientation3D()
//
// Determine orientation component of feat3D.
//
int
determineOrientation3D(
	Feature3D &feat3D
)
{
	FEATUREIO fioImg;
	fioImg.x = fioImg.z = fioImg.y = Feature3D::FEATURE_3D_DIM;
	fioImg.pfVectors = &(feat3D.data_zyx[0][0][0]);
	fioImg.pfMeans = 0;
	fioImg.pfVarrs = 0;
	fioImg.iFeaturesPerVector = 1;
	fioImg.t = 1;

	// 1st derivative images - these could be passed in from the outside.
	FEATUREIO fioDx = fioImg;
	FEATUREIO fioDy = fioImg;
	FEATUREIO fioDz = fioImg;
	Feature3D fdx;
	Feature3D fdy;
	Feature3D fdz;
	fioDx.pfVectors = &(fdx.data_zyx[0][0][0]);
	fioDy.pfVectors = &(fdy.data_zyx[0][0][0]);
	fioDz.pfVectors = &(fdz.data_zyx[0][0][0]);

	fioGenerateEdgeImages3D(fioImg, fioDx, fioDy, fioDz);

	float fMat[3][3] = { { 0,0,0 },{ 0,0,0 },{ 0,0,0 } };

	float fRadius = fioImg.x / 2;
	float fRadiusSqr = (fioImg.x / 2)*(fioImg.x / 2);

	int iSampleCount = 0;

	// Determine dominant orientation of feature: azimuth/elevation
	for (int zz = 0; zz < fioImg.z; zz++)
	{
		for (int yy = 0; yy < fioImg.y; yy++)
		{
			for (int xx = 0; xx < fioImg.x; xx++)
			{
				float dz = zz - fioImg.z / 2;
				float dy = yy - fioImg.y / 2;
				float dx = xx - fioImg.x / 2;
				if (dz*dz + dy*dy + dx*dx < fRadiusSqr)
				{
					// Keep this sample
					float pfEdge[3];
					pfEdge[0] = fioGetPixel(fioDx, xx, yy, zz);
					pfEdge[1] = fioGetPixel(fioDy, xx, yy, zz);
					pfEdge[2] = fioGetPixel(fioDz, xx, yy, zz);

					for (int i = 0; i < 3; i++)
					{
						for (int j = 0; j < 3; j++)
						{
							fMat[i][j] += pfEdge[i] * pfEdge[j];
						}
					}
				}
			}
		}
	}
	SingularValueDecomp<float, 3, 3>(fMat, feat3D.eigs, feat3D.ori);
	SortEigenDecomp<float, 3>(feat3D.eigs, feat3D.ori);

	return 0;
}

//
// sampleImage3D()
//
// Need to know the scale difference between
//
int
sampleImage3D(
	Feature3D &feat3D,
	FEATUREIO &fioSample, // 5x5x5 = 125 feats?,
	FEATUREIO &fioImg, // Original image
	FEATUREIO &fioDx, // Derivative images
	FEATUREIO &fioDy,
	FEATUREIO &fioDz
)
{
	float fOriInv[3][3];

	// We want to sample the image in a manner proportional to
	// feat3D.scale

	// Image region is 2x the feature scale
	float fImageRad = 2.0f*feat3D.scale;
	int iRadMax = fImageRad + 2;

	if (
		feat3D.x - iRadMax < 0 ||
		feat3D.y - iRadMax < 0 ||
		feat3D.z - iRadMax < 0 ||
		feat3D.x + iRadMax >= fioImg.x ||
		feat3D.y + iRadMax >= fioImg.y ||
		feat3D.z + iRadMax >= fioImg.z)
	{
		// Feature out of image bounds
		return -1;
	}

	// Invert orientation marix
	invert_3x3<float, double>(feat3D.ori, fOriInv);

	//
	// Sample pixels: trilinear interpolation
	// Here, (x,y,z) are in the sampled image coordinate system
	//
	int iSampleRad = Feature3D::FEATURE_3D_DIM / 2;
	for (int z = -iSampleRad; z <= iSampleRad; z++)
	{
		int iZ = feat3D.z + z;
		for (int y = -iSampleRad; y <= iSampleRad; y++)
		{
			int iY = feat3D.y + y;
			for (int x = -iSampleRad; x <= iSampleRad; x++)
			{
				int iX = feat3D.x + x;
				{
					// 1) Rotate feature coordinate to image coordinate

					float xyz_feat[3];  // Feat coords
					float xyz_img[3]; // Image coords
					xyz_feat[0] = x;
					xyz_feat[1] = y;
					xyz_feat[2] = z;

					mult_3x3<float, double>(fOriInv, xyz_feat, xyz_img);

					// 2) Scale feature magnitude
					float fScale = fImageRad / (float)(iSampleRad);
					xyz_img[0] *= fScale;
					xyz_img[1] *= fScale;
					xyz_img[2] *= fScale;

					// 3) Translated to current feature center
					xyz_img[0] += feat3D.x;
					xyz_img[1] += feat3D.y;
					xyz_img[2] += feat3D.z;

					// 4) Interpolate pixel
					float fPixel;

					if (xyz_img[0] < 0 || xyz_img[0] >= fioImg.x ||
						xyz_img[0] < 0 || xyz_img[0] >= fioImg.x ||
						xyz_img[0] < 0 || xyz_img[0] >= fioImg.x)
					{
						// Out of image gets black pixel (MR imagery)
						fPixel = 0;
					}
					else
					{
						// Aha - I believe we are introducing a 0.5 pixel shift backwards here
						// In GenerateFeatures3D() is corrected tho, by adding 0.5
						fPixel = fioGetPixelTrilinearInterp(
							fioImg, xyz_img[0], xyz_img[1], xyz_img[2]);
					}

					// 5) Set pixel into sample volume - could just as well set directly
					// into feature.
					float *pfVec = fioGetVector(fioSample,
						x + iSampleRad,
						y + iSampleRad,
						z + iSampleRad);
					*pfVec = fPixel;
				}
			}
		}
	}
	return 0;
}

//
// determineCanonicalOrientation3D()
//
// Determine canonical orientation of a 3D feature patch. Use cube method, to avoid angular estimation problems.
// Estimate N primary rotations, then for each, estimate the strongest secondary rotation.
//
int
determineCanonicalOrientation3D(
	Feature3D &feat3D,
	float *pfOri, // Array to store iMaxOri 3D rotation arrays. Each rotation array is
				  //			encoded as three 3D unit vectors
	int iMaxOri // Number of 6-float values available in pfOri
)
{

	FEATUREIO fioImg;
	fioImg.x = fioImg.z = fioImg.y = Feature3D::FEATURE_3D_DIM;
	fioImg.pfVectors = &(feat3D.data_zyx[0][0][0]);
	fioImg.pfMeans = 0;
	fioImg.pfVarrs = 0;
	fioImg.iFeaturesPerVector = 1;
	fioImg.t = 1;
	// 1st derivative images - these could be passed in from the outside.
	FEATUREIO fioDx = fioImg;
	FEATUREIO fioDy = fioImg;
	FEATUREIO fioDz = fioImg;
	Feature3D fdx;
	Feature3D fdy;
	Feature3D fdz;
	fioDx.pfVectors = &(fdx.data_zyx[0][0][0]);
	fioDy.pfVectors = &(fdy.data_zyx[0][0][0]);
	fioDz.pfVectors = &(fdz.data_zyx[0][0][0]);

	// Angles and blurred versions
	FEATUREIO fioT0 = fioImg;
	FEATUREIO fioT1 = fioImg;
	FEATUREIO fioT2 = fioImg;
	Feature3D fT0;
	Feature3D fT1;
	Feature3D fT2;

	fioT0.pfVectors = &(fT0.data_zyx[0][0][0]);

	fioT1.pfVectors = &(fT1.data_zyx[0][0][0]);

	fioT2.pfVectors = &(fT2.data_zyx[0][0][0]);

	fioSet(fioT0, 0);

	// Major peaks
	LOCATION_VALUE_XYZ_ARRAY lvaPeaks;
	LOCATION_VALUE_XYZ lvData[ORIBINS*ORIBINS / 8];
	float pfOriData[ORIBINS*ORIBINS / 8];
	lvaPeaks.plvz = &(lvData[0]);

	fioGenerateEdgeImages3D(fioImg, fioDx, fioDy, fioDz);


	float fRadius = fioImg.x / 2;
	float fRadiusSqr = (fioImg.x / 2)*(fioImg.x / 2);

	int iSampleCount = 0;

	// Determine dominant orientation of feature: azimuth/elevation
	for (int zz = 0; zz < fioImg.z; zz++)
	{
		for (int yy = 0; yy < fioImg.y; yy++)
		{
			for (int xx = 0; xx < fioImg.x; xx++)
			{
				float dz = zz - fioImg.z / 2;
				float dy = yy - fioImg.y / 2;
				float dx = xx - fioImg.x / 2;
				if (dz*dz + dy*dy + dx*dx < fRadiusSqr)
				{
					// Keep this sample
					float pfEdge[3];
					pfEdge[0] = fioGetPixel(fioDx, xx, yy, zz);
					pfEdge[1] = fioGetPixel(fioDy, xx, yy, zz);
					pfEdge[2] = fioGetPixel(fioDz, xx, yy, zz);
					float fEdgeMagSqr = pfEdge[0] * pfEdge[0] + pfEdge[1] * pfEdge[1] + pfEdge[2] * pfEdge[2];
					if (fEdgeMagSqr == 0)
					{
						continue;
					}
					float fEdgeMag = sqrt(fEdgeMagSqr);

					iSampleCount++;

					// Vector length fRadius in direction of edge
					float pfEdgeUnit[3];
					for (int i = 0; i < 3; i++)
					{
						pfEdgeUnit[i] = pfEdge[i] * fRadius / fEdgeMag;
					}
					for (int i = 0; i < 3; i++)
					{
						pfEdgeUnit[i] += fRadius;
					}

					fioIncPixelTrilinearInterp(fioT0, pfEdgeUnit[0] + 0.5, pfEdgeUnit[1] + 0.5, pfEdgeUnit[2] + 0.5, 0, fEdgeMag);
				}
			}
		}
	}

	iSampleCount = 0;
	for (int i = 0; i < 11 * 11 * 11; i++)
	{
		if (fioT0.pfVectors[i] > 0)
		{
			iSampleCount++;
		}
	}

	PpImage ppImgRef;
	char pcFileName[400];

	int iCode = 1;

	sprintf(pcFileName, "bin%5.5d_dx", iCode);
	sprintf(pcFileName, "bin%5.5d_dy", iCode);
	sprintf(pcFileName, "bin%5.5d_dz", iCode);

	// Could be used to cuda blur (but too long for small image)
	/*
	int iDataSizeFloat = fioT0.x*fioT0.y*fioT0.z*sizeof(float);
	cudaMalloc((float**)&fioT0.d_pfVectors, iDataSizeFloat);
	cudaMalloc((float**)&fioT2.d_pfVectors, iDataSizeFloat);
	float *array_h=static_cast<float *>(fioT0.pfVectors);//get a 1d array of pixel float
	cudaMemcpy(fioT0.d_pfVectors, array_h, iDataSizeFloat, cudaMemcpyHostToDevice); //get the array to device image
	float *array_h2=static_cast<float *>(fioT2.pfVectors);//get a 1d array of pixel float
	cudaMemcpy(fioT2.d_pfVectors, array_h2, iDataSizeFloat, cudaMemcpyHostToDevice); //get the array to device image
	*/
	gb3d_blur3d(fioT0, fioT1, fioT2, fBlurGradOriHist, 0.01, -1);
	regFindFEATUREIOPeaks(lvaPeaks, fioT2);
	lvSortHighLow(lvaPeaks);
	sprintf(pcFileName, "bin%5.5d_orig", iCode);

	sprintf(pcFileName, "bin%5.5d_blur", iCode);

	float pfP1[3]; //z
	float pfP2[3]; //y
	float pfP3[3]; //x

				   // determine sub-pixel orientation vectors
	for (int i = 0; i < lvaPeaks.iCount && i < fioImg.z && i < iMaxOri; i++)
	{
		float *pfOriCurr = &pfOriData[i * 3];

		// Interpolate
		interpolate_discrete_3D_point(fioT2,
			lvaPeaks.plvz[i].x, lvaPeaks.plvz[i].y, lvaPeaks.plvz[i].z,
			pfOriCurr[0], pfOriCurr[1], pfOriCurr[2]);

		// Subtract radius/image center of fioT2
		pfOriCurr[0] -= fRadius;
		pfOriCurr[1] -= fRadius;
		pfOriCurr[2] -= fRadius;

		// Normalize to unit length
		vec3D_norm_3d(pfOriCurr);
	}

	int iOrientationsReturned = 0;

	// Set my descriptor to 0
	fioSet(fioImg, 0);
	LOCATION_VALUE_XYZ_ARRAY lvaPeaks2;
	LOCATION_VALUE_XYZ lvData2[ORIBINS*ORIBINS / 8];
	lvaPeaks2.plvz = &(lvData2[0]);
	for (int i = 0; i < lvaPeaks.iCount && i < fioImg.z && iOrientationsReturned < iMaxOri; i++)
	{
		if (lvaPeaks.plvz[i].fValue < 0.8*lvaPeaks.plvz[0].fValue)
		{
			// Must be above threshold
			// 0.5 used in preliminary matching experiments
			// 0.8 works well to, 0.2 is terrible...
			break;
		}

		// Primary orientation unit vector
		// Used interpolated vectors here for correctnewss

		float *pfOriCurr = &pfOriData[i * 3];
		pfP1[0] = pfOriCurr[0];
		pfP1[1] = pfOriCurr[1];
		pfP1[2] = pfOriCurr[2];

		// Compute secondary direction unit vector
		// perpendicular to primary

		fioSet(fioT0, 0);
		for (int zz = 0; zz < fioImg.z; zz++)
		{
			for (int yy = 0; yy < fioImg.y; yy++)
			{
				for (int xx = 0; xx < fioImg.x; xx++)
				{
					float dx = xx - fioImg.x / 2;
					float dy = yy - fioImg.y / 2;
					float dz = zz - fioImg.z / 2;
					float fLocRadiusSqr = dz*dz + dy*dy + dx*dx;
					if (fLocRadiusSqr < fRadiusSqr)
					{
						// Vector: intensity edge
						float pfEdge[3];
						float pfEdgeUnit[3];
						pfEdge[0] = fioGetPixel(fioDx, xx, yy, zz);
						pfEdge[1] = fioGetPixel(fioDy, xx, yy, zz);
						pfEdge[2] = fioGetPixel(fioDz, xx, yy, zz);
						float fEdgeMag = vec3D_mag(pfEdge);
						if (fEdgeMag == 0)
						{
							continue;
						}
						memcpy(pfEdgeUnit, pfEdge, sizeof(pfEdgeUnit));
						vec3D_norm_3d(pfEdgeUnit);

						// Remove component parallel to primary orientatoin
						float pVecPerp[3];
						float fParallelMag = vec3D_dot_3d(pfP1, pfEdgeUnit);
						pVecPerp[0] = pfEdgeUnit[0] - fParallelMag*pfP1[0];
						pVecPerp[1] = pfEdgeUnit[1] - fParallelMag*pfP1[1];
						pVecPerp[2] = pfEdgeUnit[2] - fParallelMag*pfP1[2];

						// Normalize
						vec3D_norm_3d(pVecPerp);

						// Vector length fRadius in direction of edge, centered in image
						for (int i = 0; i < 3; i++)
						{
							pVecPerp[i] *= fRadius;
							pVecPerp[i] += fRadius;
						}

						fioIncPixelTrilinearInterp(fioT0, pVecPerp[0] + 0.5, pVecPerp[1] + 0.5, pVecPerp[2] + 0.5, 0, fEdgeMag);

					}
				}
			}
		}
		// Could be used to cuda blur (but too long for small image)
		/*
		int iDataSizeFloat = fioT0.x*fioT0.y*fioT0.z*sizeof(float);
		cudaFree(fioT0.d_pfVectors);
		cudaFree(fioT2.d_pfVectors);
		cudaMalloc((float**)&fioT0.d_pfVectors, iDataSizeFloat);
		cudaMalloc((float**)&fioT2.d_pfVectors, iDataSizeFloat);
		float *array_h=static_cast<float *>(fioT0.pfVectors);//get a 1d array of pixel float
		cudaMemcpy(fioT0.d_pfVectors, array_h, iDataSizeFloat, cudaMemcpyHostToDevice); //get the array to device image
		float *array_h2=static_cast<float *>(fioT2.pfVectors);//get a 1d array of pixel float
		cudaMemcpy(fioT2.d_pfVectors, array_h2, iDataSizeFloat, cudaMemcpyHostToDevice); //get the array to device image
		*/

		// Blur, find peaks
		gb3d_blur3d(fioT0, fioT1, fioT2, fBlurGradOriHist, 0.01, -1);
		regFindFEATUREIOPeaks(lvaPeaks2, fioT2);
		lvSortHighLow(lvaPeaks2);
		sprintf(pcFileName, "bin%5.5d_orig", iCode);

		// determine sub-pixel orientation vectors
		for (int j = 0; j < lvaPeaks2.iCount && iOrientationsReturned < fioImg.z && iOrientationsReturned < iMaxOri; j++)
		{
			if (lvaPeaks2.plvz[j].fValue < fHist2ndPeakThreshold*lvaPeaks2.plvz[0].fValue)
			{
				// fHist2ndPeakThreshold = 0.5 works well
				// Must be above threshold
				break;
			}

			// 2nd dominant orientation found, keep only first
			pfP2[0] = lvaPeaks2.plvz[j].x - fRadius;
			pfP2[1] = lvaPeaks2.plvz[j].y - fRadius;
			pfP2[2] = lvaPeaks2.plvz[j].z - fRadius;
			vec3D_norm_3d(pfP2);

			// Interpolate
			interpolate_discrete_3D_point(fioT2,
				lvaPeaks2.plvz[j].x, lvaPeaks2.plvz[j].y, lvaPeaks2.plvz[j].z,
				pfP2[0], pfP2[1], pfP2[2]);

			// Subtract radius/image center of fioT2
			pfP2[0] -= fRadius;
			pfP2[1] -= fRadius;
			pfP2[2] -= fRadius;

			// Normalize to unit length
			vec3D_norm_3d(pfP2);

			// Enforce to be perpendicular to pfP1
			float pVecPerp[3];
			float fParallelMag = vec3D_dot_3d(pfP1, pfP2);
			assert(fabs(fParallelMag) < 0.5f);
			pfP2[0] = pfP2[0] - fParallelMag*pfP1[0];
			pfP2[1] = pfP2[1] - fParallelMag*pfP1[1];
			pfP2[2] = pfP2[2] - fParallelMag*pfP1[2];

			// Re-normalize to unit length
			vec3D_norm_3d(pfP2);

			// Ensure

			// 3rd dominant orientation is cross product
			vec3D_cross_3d(pfP1, pfP2, pfP3);

			// Save vectors
			// *Note* orientation vectors saved along columns, not rows :p :p :p
			float *pfOriMatrix = pfOri + 9 * iOrientationsReturned;
			for (int ivec = 0; ivec < 3; ivec++)
			{
				pfOriMatrix[0 * 3 + ivec] = pfP1[ivec];
				pfOriMatrix[1 * 3 + ivec] = pfP2[ivec];
				pfOriMatrix[2 * 3 + ivec] = pfP3[ivec];
			}

			iOrientationsReturned++;
		}
	}
	// return - this is just for testing
	return iOrientationsReturned;
}

void
vec3D_cross_3d(
	float *pv1,
	float *pv2,
	float *pvCross
)
{
	pvCross[0] = pv1[1] * pv2[2] - pv1[2] * pv2[1];
	pvCross[1] = -pv1[0] * pv2[2] + pv1[2] * pv2[0];
	pvCross[2] = pv1[0] * pv2[1] - pv1[1] * pv2[0];
}

//
// similarity_transform_invert()
//
// Invert a similarity transform.
//
void
similarity_transform_invert(
	float *pfCenter0,
	float *pfCenter1,
	float *rot01,
	float &fScaleDiff
)
{
	float pfP0[3];

	// Exchange centroids
	memcpy(pfP0, pfCenter0, sizeof(pfP0));
	memcpy(pfCenter0, pfCenter1, sizeof(pfP0));
	memcpy(pfCenter1, pfP0, sizeof(pfP0));

	// Invert scale
	fScaleDiff = 1.0f / fScaleDiff;

	// Transform orientation/rotation matrix
	float rot01_copy[3][3];
	float rot01_trans[3][3];
	memcpy(&(rot01_copy[0][0]), &(rot01[0]), sizeof(rot01_copy));
	transpose_3x3<float, float>(rot01_copy, rot01_trans);
	memcpy(&(rot01[0]), &(rot01_trans[0][0]), sizeof(rot01_copy));
}

//
// similarity_transform_3point()
//
// Transform a point pfP0 from image 1 to a point pfP1 in image 2
// via similarity transform.
//
int
similarity_transform_3point(
	float *pfP0,
	float *pfP1,

	float *pfCenter0,
	float *pfCenter1,
	float *rot01,
	float fScaleDiff
)
{
	// Subtract origin from point
	float pfP0Diff[3];
	vec3D_diff_3d(pfCenter0, pfP0, pfP0Diff);

	// Rotate from image 0 to image 1
	float rot_one[3][3];
	memcpy(&(rot_one[0][0]), rot01, sizeof(float) * 3 * 3);
	mult_3x3<float, float>(rot_one, pfP0Diff, pfP1);

	// Scale
	pfP1[0] *= fScaleDiff;
	pfP1[1] *= fScaleDiff;
	pfP1[2] *= fScaleDiff;

	// Add to origin
	vec3D_summ_3d(pfP1, pfCenter1, pfP1);

	return 0;
}

void
vec3D_diff_3d(
	float *pv1,
	float *pv2,
	float *pv12
)
{
	pv12[0] = pv2[0] - pv1[0];
	pv12[1] = pv2[1] - pv1[1];
	pv12[2] = pv2[2] - pv1[2];
}

void
vec3D_summ_3d(
	float *pv1,
	float *pv2,
	float *pv12
)
{
	pv12[0] = pv2[0] + pv1[0];
	pv12[1] = pv2[1] + pv1[1];
	pv12[2] = pv2[2] + pv1[2];
}

//
// _sortAscendingMVNature()
//
// Break sorting ties according to natural order of SIFT descriptor values.
//
int
_sortAscendingMVNature(
	const void *pf1,
	const void *pf2
)
{
	COMP_MATCH_VALUE &f1 = *((COMP_MATCH_VALUE*)pf1);
	COMP_MATCH_VALUE &f2 = *((COMP_MATCH_VALUE*)pf2);
	if (f1.fValue < f2.fValue)
	{
		return -1;
	}
	else if (f1.fValue > f2.fValue)
	{
		return 1;
	}
	else
	{
		// Split ties in terms of index
		if (f1.iIndex1 < f2.iIndex1)
		{
			return -1;
		}
		else
		{
			return 1;
		}
	}
}
