
#ifndef __GAUSSIANMASK_H__
#define __GAUSSIANMASK_H__

#include "PpImage.h"
#include "FeatureIO.h"

//
// calculate_gaussian_filter_size()
//
// Calculates the size of a Gaussian filter in 1 dimension,
// such that the cut-off of the Gaussian is less than or equal
// to fMinValue multiplied by the maximum value of the Gaussian (center)
//
int
calculate_gaussian_filter_size(
							   const float &fSigma,
							   const float &fMinValue
							   );


//
// generate_gaussian_filter2d()
//
// Generates a 2d gaussian pattern specified by the given parameters.
//
//	ppTmp - output image, data type float (32-bits per pixel)
//	fSigmaRow - standard deviation of row gaussian component
//	fSigmaCol - standard deviation of col gaussian component
//	fMeanRow - mean of row gaussian component
//	fMeanCol - mean of col gaussian component
//
int
generate_gaussian_filter2d(
	PpImage		&ppTmp,
	const float &fSigmaRow,
	const float &fSigmaCol,
	const float &fMeanRow,
	const float &fMeanCol
					 );
//
// generate_gaussian_filter2d()
//
// Generates a 2d gaussian pattern specified by the given parameters.
//
//	ppTmp - output image, data type float (32-bits per pixel)
//	fSigmaRow - standard deviation of row gaussian component
//	fSigmaCol - standard deviation of col gaussian component
//	fTheta    - rotation about the mean row / mean col
//	fMeanRow - mean of row gaussian component
//	fMeanCol - mean of col gaussian component
//
int
generate_gaussian_filter2d(
	PpImage		&ppTmp,
	const float &fSigmaRow,
	const float &fSigmaCol,
	const float &fTheta,
	const float &fMeanRow,
	const float &fMeanCol
					 );


//
// generate_gaussian_filter2d()
//
// Generates a 2d gaussian pattern specified by the given parameters.
//
//	ppTmp - output image, data type float (32-bits per pixel)
//	fSigmaRow - standard deviation of row gaussian component
//	fSigmaCol - standard deviation of col gaussian component
//	fTheta    - rotation about the mean row / mean col
//	fTheta    - derivative rotation about the mean row / mean col
//	fMeanRow - mean of row gaussian component
//	fMeanCol - mean of col gaussian component
//
int
generate_gaussian_filter2d_derivative(
	PpImage		&ppTmp,
	const float &fSigmaRow,
	const float &fSigmaCol,
	const float &fTheta,
	const float &fDerivativeTheta,
	const float &fMeanRow,
	const float &fMeanCol
					 );

//
// generate_gaussian_filter1d()
//
// Generates a 1D Gaussian filter, ppTmp has 1 row, several columns.
//
int
generate_gaussian_filter1d(
	PpImage		&ppTmp,
	const float &fSigmaCol,
	const float &fMeanCol
					 );

int
generate_gaussian_filter1d(
    float *pfFilter,
	int iSize,
	const float &fSigmaCol,
	const float &fMeanCol
	);

	//
	// generate_gaussian_filter3d()
	//
	// Generates a 3D Gaussian filter, ppTmp has 1 row, several columns.
	//

int
generate_gaussian_filter3d(
	PpImage		&ppTmp,
	const float &fSigmaCol,
	const float &fMeanCol
);

#endif
