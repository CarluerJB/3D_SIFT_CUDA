
//
// File: FeatureIO.h
// Desc: Interface for a feature IO system, managing 2D and 3D arrays
//	of feature vectors. Arrays are stored in two corresponding
//	files *.bin and *.txt
//
//	*.bin: binary feature vector data, stored as interleaved floating points.
//
//	*.txt: text description of feature data.
//
//

#ifndef __FEATUREIO_H__
#define __FEATUREIO_H__

#define DIM_Z 1
#define DIM_Y 2
#define DIM_X 3

typedef struct _FEATUREIO
{
	int z;
	int y;
	int x;
	int t;
	int iFeaturesPerVector;
	float *pfVectors;
	float *d_pfVectors; // pointer to data on GPU
	float *pfMeans;	// Mean
	float *pfVarrs; // Varriance
} FEATUREIO;

typedef struct _OUTPUT_RANGE
{
	float fMin;
	float fMax;
	int bClipToValues;
} OUTPUT_RANGE;

//
// fioRead()
//
// Read files associated with features.
//
int
fioRead(
		FEATUREIO &fio,
		char *pcName
		);

//
//
// Reads a raw binary 32-bit float file.
//
int
fioReadRaw(
				 FEATUREIO &fio,
				 char *pcFileName
				 );

//
// fioReadAllInOne()
//
// Reads a text file with all FEATUREIO information,
// including the name of the associated data file.
//
int
fioReadAllInOne(
		FEATUREIO &fio,
		char *pcName
		);

//
// fioAllocate()
//
// Allocates memory for a FEATUREIO struct.
//
int
fioAllocate(
		FEATUREIO &fio
		);

//
// fioAllocateExample()
//
// Initialize fio to be the same size as fioExample.
//
int
fioAllocateExample(
		FEATUREIO &fio,
		const FEATUREIO &fioExample
		);

//
// fioDelete()
//
// Delete memory allocated for features.
//
int
fioDelete(
		FEATUREIO &fio
		);


//
// fioTraverseFIO()
//
// For each point in fio, call func( t, x, y, z, pData ). To
// keep going, func() should return 0. Other
//
typedef int POINT_FUNC( int t, int x, int y, int z, void *pData );

//
//
int
fioTraverseFIO(
			   FEATUREIO &fio,
			   POINT_FUNC func,
			   void *pData
			   );

//
// fioModify()
//
// Removes, and performs a difference.
//
int
fioModify(
		FEATUREIO &fio,
		int *piRemove,		// List relative indices for replacement
		int bDiff			//
		);

//
// fioEstimateVarriance()
//
// Estimates the varriance of features.
//
int
fioEstimateVarriance(
					 FEATUREIO &fio
					 );

//
// fioSet()
//
// Sets all features to a certain value
//
int
fioSet(
		FEATUREIO &fio,
		float fValue
		);

float
fioGetPixel(
		const FEATUREIO &fio,
		int x, int y, int z=0, int t=0
			);

float *
fioGetVector(
		const FEATUREIO &fio,
		int x, int y, int z=0, int t=0
			);

//
// fioGetPixelTrilinearInterp()
//
// fioGetPixelBilinearInterp()
//
// Performs bilinear/trilinear interpolation.
//
// Assumes the center of pixel 0,0,0 is 0.5,0.5,0.5,
// so add 0.5 to all coordinates.
//
float
fioGetPixelBilinearInterp(
						   const FEATUREIO &fio,
						   float x, float y
						   );

float
fioGetPixelTrilinearInterp(
						   const FEATUREIO &fio,
						   float x, float y, float z
						   );

//
// Functions to either set or increment via subpixel interpolation values.
// Add/set a fraction of fValue to neighboring bins.
//
void
fioSetPixelTrilinearInterp(
						   const FEATUREIO &fio,
						   float x, float y, float z,
						   int iFeature, float fValue
						   );

float
fioGetPixelTrilinearInterp(
						   const FEATUREIO &fio,
						   float x, float y, float z, int iFeature
						   );
void
fioIncPixelTrilinearInterp(
						   const FEATUREIO &fio,
						   float x, float y, float z,
						   int iFeature, float fValue
						   );

//
// fioWriteSlice()
//
// Copies an image slice into a 2D array
//
int
fioFeatureSliceXY(
		FEATUREIO &fio,
		int iValue,			// Value in Z dimension
		int iFeature,		// Index of feature to output
		float *pfSlice		// Memory for slice
		);
int
fioFeatureSliceZX(
		FEATUREIO &fio,
		int iValue,			// Value in Z dimension
		int iFeature,		// Index of feature to output
		float *pfSlice		// Memory for slice
		);
int
fioFeatureSliceZY(
		FEATUREIO &fio,
		int iValue,			// Value in Z dimension
		int iFeature,		// Index of feature to output
		float *pfSlice		// Memory for slice
		);

//
// fioExtractFeature()
//
// Extracts a single feature image from a multi-featured
// array. Copies feature iFeature from fioFeatureMulti
// to fioFeatureSingle.
//
int
fioExtractSingleFeature(
		FEATUREIO &fioFeatureSingle,
		FEATUREIO &fioFeatureMulti,
		int iFeature
		);

//
// fioCrop()
// Crops fioIn, from the point (xin,yin,zin) to
// the size of fioOut, stores the result in fioOut. fioOut
// should be preallocated.
//
int
fioCrop(
		FEATUREIO &fioIn,
		FEATUREIO &fioOut,
		int xin,
		int yin,
		int zin
		);

//
// fioWrite()
//
// Write features to associated files.
//
int
fioWrite(
		FEATUREIO &fio,
		char *pcName
		);


//
// fioZero()
//
// Set all fields to 0
//
int
fioZero(
		FEATUREIO &fio
		);

//
// fioFindMinMax()
//
// Find max/min of image.
//
int
fioFindMinMax(
		FEATUREIO &fio,
			 float &fMin,			// Min value
			 float &fMax				// Max value
			 );


//
// fioNormalizeSquaredLength()
//
// Normalizes fio to unit squared length.
//
int
fioNormalizeSquaredLength(
						  FEATUREIO &fio
						  );


int
fioNormalizeSquaredLengthZeroMean(
						  FEATUREIO &fio
						  );

//
// fioNormalize()
//
// Normalizes feature intensities, and stretches to a range of 0 to fStretch.
// Only works on feature vectors with 1 feature per vector.
//
int
fioNormalize(
		FEATUREIO &fio,
		const float &fStretch
		);

// fioSubSampleInterpolate()
//
// Subsamples the feature IO file.  Stores the result in fioOut. fioOut
// should be preallocated and have half the dimensions as fioIn.
// Same as fioSubSample except bilinear interpolation is performed.
//
int
fioSubSampleInterpolate(
		FEATUREIO &fioIn,
		FEATUREIO &fioOut
		);

// Subsample_interleave()
//
// Call cuda or cpu Subsamples function
//
//
//
int
Subsample_interleave(
	FEATUREIO &fioIn,
	FEATUREIO &fioOut,
	int best_device_id);


// fioSubSample()
//
// Subsamples the feature IO file.  Stores the result in fioOut. fioOut
// should be preallocated and have half the dimensions as fioIn.
//
int
fioSubSample(
		FEATUREIO &fioIn,
		FEATUREIO &fioOut
		);

// fioSubSample2D()
//
// Subsamples a 2D feature IO file.  Stores the result in fioOut. fioOut
// should be preallocated and have half the dimensions (in x and y) as fioIn.
//
int
fioSubSample2D(
		FEATUREIO &fioIn,
		FEATUREIO &fioOut
		);

// fioSubSample2D()
//
// Subsamples a 2D feature IO file.  Stores the result in fioOut. fioOut
// should be preallocated and have half the dimensions (in x and y) as fioIn.
//
// Each pixel in the output image is generated from an even bilinear sample
// of the input image, thus the image remains 'centered'. This makes the output
// image a little smoother, and makes it easier to calculate the coordinates
// of the output image wrt to input image.
//
int
fioSubSample2DCenterPixel(
		FEATUREIO &fioIn,
		FEATUREIO &fioOut
		);

//
//
//
//
//
int
fioMultiply(
		FEATUREIO &fioIn1,
		FEATUREIO &fioIn2,
		FEATUREIO &fioOut
		);

//
// fioCopy()
//
// Copies fioSrc to fioDst.
//
int
fioCopy(
		FEATUREIO &fioDst,
		FEATUREIO &fioSrc
		);

//
// fioCopy()
//
// Copies one feature to another
//
int
fioCopy(
		FEATUREIO &fioDst,
		FEATUREIO &fioSrc,
		int iDstFeature,
		int iSrcFeature
		);

//
// fioDimensionsEqual()
//
// Returns true if the dimensions are equal,
// false otherwise.
//
int
fioDimensionsEqual(
		FEATUREIO &fio1,
		FEATUREIO &fio2
		);

//
// fioSum()
//
// Sums up the values of feature iFeature in fio, returns the result.
//
float
fioSum(
	   FEATUREIO &fio,
	   int iFeature = 0
	   );


//
// fioAbs()
//
// Takes the absolute value of each feature.
//
int
fioAbs(
	   FEATUREIO &fio
	   );


 //
 // fioMultSum_interleave()
 //
 // Launch fioMultSum on cpu or gpu
 //

int
fioMultSum_interleave(
		FEATUREIO &fioIn1,
		FEATUREIO &fioIn2,
		FEATUREIO &fioOut,
		const float &fMultIn2,
		int best_device_id
		);


//
// fioMultSum()
//
// fioOut = fioIn1 + fMultIn2*fioIn2.
//
int
fioMultSum(
		FEATUREIO &fioIn1,
		FEATUREIO &fioIn2,
		FEATUREIO &fioOut,
		const float &fMultIn2
		);


//
// fioMin()
//
// fioOut = min( fioIn1, fioIn2 )
//
int
fioMin(
		FEATUREIO &fioIn1,
		FEATUREIO &fioIn2,
		FEATUREIO &fioOut
		);

//
// fioMult()
//
// fioOut = fMultIn*fioIn
//
int
fioMult(
		FEATUREIO &fioIn1,
		FEATUREIO &fioOut,
		const float &fMultIn
		);

//
// initNeighbourOffsets()
//
// Initializes an array of offsets of pixel nearest neighbours
// in a FEATUREIO. For a 3D FEATUREIO, piNeighbourOffsets must
// have enough space for a maximum of 26 offsets, for a 2D FEATUREIO,
// 8 offsets.
//
// If bSemetric is selected, only positive neighbours are included, i.e.
//   (z,y,x)->(z,y,x+1) and (z,y,x)->(z,y+1,x) are included,
//   (z,y,x)->(z,y,x-1) and (z,y,x)->(z,y-1,x) are not included.
//
int
fioInitNeighbourOffsets(
			   FEATUREIO &fio,
			   int &iNeighbourCount,	// Number of offsets initialized
			   int *piNeighbourOffsets,	// Array of offsets
			   int bDiagonal = 0,       // Diagonal or square nearest neighbours
			   int bSymetric = 0		// If symetric, only include half of nearest neighbours
			   );

//
// fioIndexToCoord()
//
// Computes the (x,y,z) coordinate cooresponding to a feature index
// in fio.
//
void
fioIndexToCoord(
				FEATUREIO &fio,
				const int &iIndex,
				int &x,
				int &y,
				int &z
				);

//
// fioCoordToIndex()
//
// Computes the feature index cooresponding to a (x,y,z) coordinate
// in fio.
//
void
fioCoordToIndex(
				FEATUREIO &fio,
				int &iIndex,
				const int &x,
				const int &y,
				const int &z
				);

//
// fioGenerateHarrFilters()
//
// Generates an array of orthogonal Harr wavelet filters.
// Each filter is of the size specified in fioSize, which must be
// of dimension at least (iXLevels, iYLevels, iZLevels).
//
// Ideally, fioSize.x is an even multiple of 2^iXLevels.
//
// Memory is allocated for pfioArray.
//
int
fioGenerateHarrFilters(
					   const int &iXLevels,
					   const int &iYLevels,
					   const int &iZLevels,
					   FEATUREIO fioSize,
					   FEATUREIO *& pfioArray
				);


//
// fioGenerateEdgeImages3D()
//
// Generates edge images in 3D. All images are single-feature
// and of equal dimensionality.
//
int
fioGenerateEdgeImages3D(
					  FEATUREIO &fioImg,
					  FEATUREIO &fioDx,
					  FEATUREIO &fioDy,
					  FEATUREIO &fioDz
				);


//
// fioGenerateEdgeImages()
//
// Generates basic edge images.
//
int
fioGenerateEdgeImages2D(
					  FEATUREIO &fioImg,
					  FEATUREIO &fioDx,
					  FEATUREIO &fioDy
				);

//
// fioGenerateOrientationImage()
//
// Generates an orientation image, no
//
int
fioGenerateOrientationImages2D(
							 FEATUREIO &fioImg,
							 FEATUREIO &fioMag,
							 FEATUREIO &fioOri
							 );

//
// fioFindExtrema2D()
//
// Detects extrema in a 2D image, callback with the coordinates
// and the extrema type.
//

typedef int FIND_LOCATION_FUNCTION( int x, int y, int z, int t, int iMaximum, float fValue, void *pData );

int
fioFindExtrema2D(
			   FEATUREIO &fioImg,
			   FIND_LOCATION_FUNCTION *func,
			   void *pData
			   );


//
// fioDoubleSize()
//
// Doubles image size - for feature extraction.
//
int
fioDoubleSize(
		   FEATUREIO &fio
		   );

//
// fioCreateJointImageInterleaved()
//
// Create a joint image, e.g. a joint histogram image.
//
// Huge on memory, the joint image feature size is fio1.iFeaturesPerVector * fio2.iFeaturesPerVector
//
// In the end, Gauss blur does not work on interleaved images so create a joint image array instead
// using fioCreateJointImageArray() below.
//
int
fioCreateJointImageInterleaved(
		   FEATUREIO &fioImg1,
		   FEATUREIO &fioImg2,
		   FEATUREIO &fioJoint
		   );

//
// fioInitJointImageArray
// All arrays are allocated, put fioImg1/2 joint in fioJointArray
//
int
fioInitJointImageArray(
		   FEATUREIO &fioImg1,
		   FEATUREIO &fioImg2,
		   FEATUREIO *fioJointArray // Array of FEATUREIO
		   );

//
// fioInitJointImageArray()
//
// Same as fioInitJointImageArray() above, except inputs are non-interleaved
// FEATUREIO arrays of lengths iCount1/2.
//
int
fioInitJointImageArray(
		   FEATUREIO *fioImg1,
		   FEATUREIO *fioImg2,
		   int iCount1,
		   int iCount2,
		   FEATUREIO *fioJointArray // Array of FEATUREIO
		   );

//
// fioNormalizeJointImageArray()
//
// Normalize vectors to a distribution.
//
int
fioNormalizeJointImageArray(
		   int iCount1,
		   int iCount2,
		   FEATUREIO *fioJointArray // Array of FEATUREIO
							);


int
fioFindMin(
		   FEATUREIO &fio,
		   int &ix,
		   int &iy,
		   int &iz
		   );

int
fioFindMax(
		   FEATUREIO &fio,
		   int &ix,
		   int &iy,
		   int &iz
		   );

int
fioFindMin(
		   FEATUREIO &fio,
		   int &ix,
		   int &iy,
		   int &iz,
		   float &fMin
		   );

int
fioFindMax(
		   FEATUREIO &fio,
		   int &ix,
		   int &iy,
		   int &iz,
		   float &fMax
		   );

int
fioFindMin(
		   FEATUREIO &fio,
		   float &fValue
		   );

int
fioFindMax(
		   FEATUREIO &fio,
		   float &fValue
		   );

//
// fioCalculateSVD()
//
// Calculate principal components via singular value decomposition.
// Calculate PCA via SVD.
//
void
fioCalculateSVD(
				FEATUREIO &fioIn,
				FEATUREIO &fioPCs,
				float *pfEigenValues
			   );

//
// fioTranslate()
//
// Translate image fioSrc into fioDst.
//    fioDst = fioSrc + [dx,dy,dz]
// Pixels out of image range are set to 0.
//
int
fioTranslate(
			 FEATUREIO &fioDst,
			FEATUREIO &fioSrc,
			int dx, int dy, int dz
);

int
fioPixelCount(
			 FEATUREIO &fio
			  );

//
// fioInterpolateLine()
//
// Interpolate a line of iCount samples from point 1 to point 2 in an image.
//
int
fioInterpolateLine(
				 FEATUREIO &fio,
				 float x1, float y1, float z1,
				 float x2, float y2, float z2,
				 float *pfLine,
				 int iCount
				 );

//
// fioFadeGaussianWindow()
//
// Multiply image intensities by a Gaussian of stdev fSigma centered on pfXYZ.
// Values in the perifery of the Gaussian will fade to black.
//
int
fioFadeGaussianWindow(
		FEATUREIO &fio,
		float *pfXYZ,
		float fSigma
		);

int
fioFlipAxisY(
		FEATUREIO &fio
			 );
int
fioFlipAxisX(
		FEATUREIO &fio
			 );


#endif
