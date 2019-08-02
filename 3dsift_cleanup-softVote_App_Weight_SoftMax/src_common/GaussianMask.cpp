
#include "GaussianMask.h"

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "FeatureIO.h"

#define PI				3.1415926535897932384626433832795
#define ROOT_2			1.4142135623730950488016887242097

int
calculate_gaussian_filter_size(
							   const float &fSigma,
							   const float &fMinValue
							   )
{
	float fPower = 0.0f;
	float fValue = (float)exp( fPower );
	float fMaxValue = fValue;
	int i;

	if( fSigma == 0 )
	{
		// Delta function - 1 sample
		return 1;
	}

	// Estimate Gaussian volume under curve
	float fCurVolume = 1; // exp(-0)
	float fNewVolume = 1; // exp(-0)
	i = 0;
	do
	{
		i++;
		fCurVolume = fNewVolume;
		fPower = ((float)(i*i)) / ((float)-2.0*fSigma*fSigma);
		fNewVolume = fCurVolume + 2*(float)exp( fPower );
	} while( fNewVolume - fCurVolume > 0.00001f );

	//
	//
	for( i = 1; fValue <= fCurVolume*(1.0f-fMinValue); i++ )
	{
		fPower = ((float)(i*i)) / ((float)-2.0*fSigma*fSigma);
		fValue += 2*(float)exp( fPower );
	}
	i--;
	//for( i = 1; fValue >= fMaxValue*fMinValue; i++ )
	//{
	//	fPower = ((float)(i*i)) / ((float)-2.0*fSigma*fSigma);
	//	fValue = (float)exp( fPower );
	//}

	return 2*i+1;
	//return (i+((i/2)*2));
}

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
					 )
{
	assert( fMeanRow >= 0.0 && fMeanRow < ppTmp.Rows() );
	assert( fMeanCol >= 0.0 && fMeanCol < ppTmp.Cols() );

	float fSigmaRowSqr = fSigmaRow*fSigmaRow;
	float fSigmaColSqr = fSigmaCol*fSigmaCol;

	float fScale = (float)(1.0 / (fSigmaRow*fSigmaCol*2.0*PI));

	for(int i = 0; i < ppTmp.Rows(); i++)
	{
		float fRowPos = ((float)i - fMeanRow);
		float fFactorRow = (fRowPos*fRowPos)/fSigmaRowSqr;

		float *pfImgRow = (float*)ppTmp.ImageRow( i );

		for(int j = 0; j < ppTmp.Cols(); j++)
		{
			float fColPos = ((float)j - fMeanCol);

			float fPower = (fFactorRow + (fColPos*fColPos)/fSigmaColSqr)
								/ (float)(-2.0);

			pfImgRow[ j ] = (float)(fScale*exp( fPower ));
		}
	}

	return 1;
}

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
generate_gaussian_filter2d_derivative(
	PpImage		&ppTmp,
	const float &fSigmaRow,
	const float &fSigmaCol,
	const float &fTheta,
	const float &fDerivativeTheta,
	const float &fMeanRow,
	const float &fMeanCol
					 )
{
	assert( fMeanRow >= 0.0 && fMeanRow < ppTmp.Rows() );
	assert( fMeanCol >= 0.0 && fMeanCol < ppTmp.Cols() );

	float fSigmaRowSqr = fSigmaRow*fSigmaRow;
	float fSigmaColSqr = fSigmaCol*fSigmaCol;

	float fScale = (float)(1.0 / (fSigmaRow*fSigmaCol*2.0*PI));

	float fSinTheta = (float)sin(fTheta);
	float fCosTheta = (float)cos(fTheta);

	// Unit vector in derivative direction
	float fIncRow = (float)sin(fDerivativeTheta);
	float fIncCol = (float)cos(fDerivativeTheta);

	for(int i = 0; i < ppTmp.Rows(); i++)
	{
		float fRowPos = ((float)i - fMeanRow) + fIncRow;
		float fRowNeg = ((float)i - fMeanRow) - fIncRow;

		float fNewRowPosCos = fRowPos*fCosTheta;
		float fNewRowPosSin = fRowPos*fSinTheta;

		float fNewRowNegCos = fRowNeg*fCosTheta;
		float fNewRowNegSin = fRowNeg*fSinTheta;

		float *pfImgRow = (float*)ppTmp.ImageRow( i );

		for(int j = 0; j < ppTmp.Cols(); j++)
		{
			float fColPos = ((float)j - fMeanCol) + fIncCol;
			float fColNeg = ((float)j - fMeanCol) - fIncCol;

			float fNewRowPos = fNewRowPosCos - fColPos*fSinTheta;
			float fNewColPos = fNewRowPosSin + fColPos*fCosTheta;

			float fNewRowNeg = fNewRowNegCos - fColNeg*fSinTheta;
			float fNewColNeg = fNewRowNegSin + fColNeg*fCosTheta;

			float fPowerPositive =
				((fNewRowPos*fNewRowPos)/fSigmaRowSqr + (fNewColPos*fNewColPos)/fSigmaColSqr)
				/ -2.0f;

			float fPowerNegative =
				((fNewRowNeg*fNewRowNeg)/fSigmaRowSqr + (fNewColNeg*fNewColNeg)/fSigmaColSqr)
				/ -2.0f;

			pfImgRow[ j ] = (float)(fScale*exp( fPowerPositive )) - (float)(fScale*exp( fPowerNegative ));
		}
	}

	return 1;
}

int
generate_gaussian_filter2d(
	PpImage		&ppTmp,
	const float &fSigmaRow,
	const float &fSigmaCol,
	const float &fTheta,	// This represents the covariance between row & column terms
	const float &fMeanRow,
	const float &fMeanCol
					 )
{
	assert( fMeanRow >= 0.0 && fMeanRow < ppTmp.Rows() );
	assert( fMeanCol >= 0.0 && fMeanCol < ppTmp.Cols() );

	float fSigmaRowSqr = fSigmaRow*fSigmaRow;
	float fSigmaColSqr = fSigmaCol*fSigmaCol;

	float fScale = (float)(1.0 / (fSigmaRow*fSigmaCol*2.0*PI));

	float fSinTheta = (float)sin(fTheta);
	float fCosTheta = (float)cos(fTheta);

	for(int i = 0; i < ppTmp.Rows(); i++)
	{
		float fRowPos = ((float)i - fMeanRow);

		float fNewRowCos = fRowPos*fCosTheta;
		float fNewRowSin = fRowPos*fSinTheta;

		float *pfImgRow = (float*)ppTmp.ImageRow( i );

		for(int j = 0; j < ppTmp.Cols(); j++)
		{
			float fColPos = ((float)j - fMeanCol);

			float fNewRowPos = fNewRowCos - fColPos*fSinTheta;
			float fNewColPos = fNewRowSin + fColPos*fCosTheta;

			float fFactorRow = (fNewRowPos*fNewRowPos)/fSigmaRowSqr;
			float fFactorCol = (fNewColPos*fNewColPos)/fSigmaColSqr;

			float fPower = (fFactorRow + fFactorCol);

			fPower /= -2.0f;

			pfImgRow[ j ] = (float)(fScale*exp( fPower ));
		}
	}

	return 1;
}

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
	)
{
	assert( fMeanCol >= 0.0 && fMeanCol < ppTmp.Cols() );
	float fSigmaColSqr = fSigmaCol*fSigmaCol;

	float fScale = (float)(1.0 / (fSigmaCol*sqrt(2.0*PI)));

	float *pfImgRow = (float*)ppTmp.ImageRow(0);

	for(int j = 0; j < ppTmp.Cols(); j++)
	{
		float fColPos = ((float)j - fMeanCol);

		float fPower = ((fColPos*fColPos)/fSigmaColSqr)
							/ (float)(-2.0f);

		pfImgRow[ j ] = (float)(fScale*exp( fPower ));
	}
	return 1;
}

int
generate_gaussian_filter3d(
	PpImage		&ppTmp,
	const float &fSigmaCol,
	const float &fMeanCol
	)
{
	assert( fMeanCol >= 0.0 && fMeanCol < ppTmp.Cols() );
	float fSigmaColSqr = fSigmaCol*fSigmaCol;
	float fSigmaCol3Sqr = fSigmaCol*fSigmaCol*fSigmaCol;

	float fScale = (float)(1.0 / (fSigmaCol3Sqr*pow((2.0*PI),(3/2))));

	float *pfImgRow = (float*)ppTmp.ImageRow(0);

	for(int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			for (int z = 0; z < 3; z++)
			{
				float fColPos = ((float)(x*x) + (float)(y*y) + (float)(z*z));
				float fPower = ((fColPos)/((float)2.0*fSigmaColSqr));
				int i=x+y*3+z*3*3;
				pfImgRow[ i ] = (float)(fScale*exp( fPower ));
			}
		}
	}

	return 1;
}

int
generate_gaussian_filter1d(
    float *pfFilter,
	int iSize,
	const float &fSigmaCol,
	const float &fMeanCol
	)
{
	assert( fMeanCol >= 0.0 && fMeanCol < iSize );

	float fSigmaColSqr = fSigmaCol*fSigmaCol;

	float fScale = (float)(1.0 / (fSigmaCol*sqrt(2.0*PI)));

	float *pfImgRow = pfFilter;

	for(int j = 0; j < iSize; j++)
	{
		float fColPos = ((float)j - fMeanCol);

		float fPower = ((fColPos*fColPos)/fSigmaColSqr)
							/ (float)(-2.0f);

		pfImgRow[ j ] = (float)(fScale*exp( fPower ));
	}

	return 1;
}
