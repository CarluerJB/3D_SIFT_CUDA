
//
// File: main.cpp
// Desc: Test driver for FeatureIO.(h/cpp).
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "PpImageFloatOutput.h"
#include "PpImage.h"
#include "GaussianMask.h"
#include "PpImage.h"
#include "FeatureIO.h"
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;


#include "SIFT_cuda_Tools.cuh"

#include "GaussBlur3D.h"

// OMG trying another Gaussian convolution algorithm after 10 years!!
// 2013 better late than never!
//#include "gaussian_conv_vyv.h"

//#include "gaussian_conv_deriche.h"
//#include "gaussian_conv_box.h"
//#include "gaussian_conv_ebox.h"
//#include "gaussian_conv_sii.h"

//
// filter_1d()
//
// Simple 1d filter.
//
void
filter_1d(
		  float *pfFilter,
		  float *pfBufferIn,
		  float *pfBufferOut,
		  int iFiltLen,
		  int iBufferLen
		  )
{
	for( int i = 0; i < iBufferLen - iFiltLen; i++ )
	{
		float fSum = 0;
		for( int j = 0; j < iFiltLen; j++ )
		{
			fSum += pfFilter[j]*pfBufferIn[i+j];
		}
		pfBufferOut[i+iFiltLen/2] = fSum;
	}
}

//
// blur_3d()
//
// Performs a 3D blur on an image. Assumes data is organized
// as y, x, z. Assumes black pixels at image border.
//
int
blur_3d(
		FEATUREIO &fio1,
		FEATUREIO &fioTemp,
		FEATUREIO &fio2,
		int	iFeature,
		PpImage &ppImgFilter
		)
{
	int x_dim = fio1.x;
	int y_dim = fio1.y;
	int z_dim = fio1.z;

	int yy_dim = y_dim;
	int xx_dim = x_dim;

	int len = ppImgFilter.Cols();
	int half_len = ppImgFilter.Cols() / 2;
	assert( (len%2) == 1 );
//	assert( len <= fio1.x );
//	assert( len <= fio1.y );
//	assert( len <= fio1.z );

	float *pfFilter = (float*)ppImgFilter.ImageRow(0);

	// Blur x direction, store for y
	// fio1: y,x,z
	// fio2: x,y,z

	for( int z = 0; z < z_dim; z++ )
	{
		int iPlane = z*y_dim*x_dim;

		for( int r = 0; r < yy_dim; r++ )
		{
			float *pfData = fio1.pfVectors + iPlane + r*xx_dim;
			int len = 2*half_len+1;

			if( len > xx_dim )
			{
				for( int c = 0; c < xx_dim; c++ )
				{
					int fi = ( c > half_len ? 0 : half_len - c );
					int di = ( c > half_len ? c - half_len : 0 );
					int lenback  = half_len - fi;
					int lenfrnt  = ( c + half_len < xx_dim ? half_len : xx_dim - c - 1);
					int lentotal = lenback + lenfrnt + 1;
					float fSum = 0;
					for(int i = 0 ; i < lentotal; fi++, di++, i++ )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					// Store in same plane, other image, switch rows/cols
					fio2.pfVectors[(iPlane+c*yy_dim+r)*fio2.iFeaturesPerVector + iFeature] = fSum;
				}
			}
			else
			{
				int c;
				for( c = 0; c < half_len; c++ )
				{
					int fi = half_len - c;
					int di = 0;
					float fSum = 0;
					for( ; fi < len; fi++, di++ )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					// Store in same plane, other image, switch rows/cols
					fio2.pfVectors[(iPlane+c*yy_dim+r)*fio2.iFeaturesPerVector + iFeature] = fSum;
				}

				for( ; c < xx_dim - half_len; c++ )
				{
					int fi = 0;
					int di = c - half_len;
					float fSum = 0;
					for( ; fi < len; fi++, di++ )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					// Store in same plane, other image, switch rows/cols
					fio2.pfVectors[(iPlane+c*yy_dim+r)*fio2.iFeaturesPerVector + iFeature] = fSum;
				}

				for( ; c < xx_dim; c++ )
				{
					int fi = 0;
					int di = c - half_len;
					float fSum = 0;
					len--;
					for( ; fi < len; fi++, di++ )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					// Store in same plane, other image, switch rows/cols
					fio2.pfVectors[(iPlane+c*yy_dim+r)*fio2.iFeaturesPerVector + iFeature] = fSum;
				}
			}
		}
	}

	// Blur y direction, store for z
	// fio2: x,y,z
	// fioTemp: z,x,y

	for( int z = 0; z < z_dim; z++ )
	{
		int iPlane = z*xx_dim*yy_dim;

		for( int r = 0; r < xx_dim; r++ )
		{
			float *pfData = fio2.pfVectors + (iPlane + r*yy_dim)*fio2.iFeaturesPerVector;
			int len = 2*half_len+1;

			if( len > fio1.y )
			{
				for( int c = 0; c < y_dim; c++ )
				{
					int fi = ( c > half_len ? 0 : half_len - c );
					int di = ( c > half_len ? c - half_len : 0 );
					int lenback  = half_len - fi;
					int lenfrnt  = ( c + half_len < xx_dim ? half_len : xx_dim - c - 1);
					int lentotal = lenback + lenfrnt + 1;
					float fSum = 0;
					for( int i = 0; i < lentotal; fi++, di+=fio2.iFeaturesPerVector, i++ )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					// Store in same plane, other image, switch rows/cols
					fioTemp.pfVectors[r*z_dim*yy_dim + c*z_dim + z] = fSum;
				}
			}
			else
			{
				int c;
				for( c = 0; c < half_len; c++ )
				{
					int fi = half_len - c;
					int di = (0)*fio2.iFeaturesPerVector + iFeature;
					float fSum = 0;
					for( ; fi < len; fi++, di+=fio2.iFeaturesPerVector )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					// Store in same plane, other image, switch rows/cols
					fioTemp.pfVectors[r*z_dim*yy_dim + c*z_dim + z] = fSum;
				}

				for( ; c < yy_dim - half_len; c++ )
				{
					int fi = 0;
					int di = (c - half_len)*fio2.iFeaturesPerVector + iFeature;
					float fSum = 0;
					for( ; fi < len; fi++, di+=fio2.iFeaturesPerVector )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					// Store in same plane, other image, switch rows/cols
					fioTemp.pfVectors[r*z_dim*yy_dim + c*z_dim + z] = fSum;
				}

				for( ; c < yy_dim; c++ )
				{
					int fi = 0;
					int di = (c - half_len)*fio2.iFeaturesPerVector + iFeature;
					float fSum = 0;
					len--;
					for( ; fi < len; fi++, di+=fio2.iFeaturesPerVector )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					// Store in same plane, other image, switch rows/cols
					fioTemp.pfVectors[r*z_dim*yy_dim + c*z_dim + z] = fSum;
				}
			}
		}
	}

	// Blur z direction, store as input
	// fioTemp: z,x,y
	// fio2: y,x,z

	for( int y = 0; y < xx_dim; y++ )
	{
		int iPlane = y*z_dim*yy_dim;

		for( int r = 0; r < yy_dim; r++ )
		{
			float *pfData = fioTemp.pfVectors + iPlane + r*z_dim;
			int len = 2*half_len+1;

			if( len > fio1.z )
			{
				for( int c = 0; c < z_dim; c++ )
				{
					int fi = ( c > half_len ? 0 : half_len - c );
					int di = ( c > half_len ? c - half_len : 0 );
					int lenback  = half_len - fi;
					int lenfrnt  = ( c + half_len < xx_dim ? half_len : xx_dim - c - 1);
					int lentotal = lenback + lenfrnt + 1;
					float fSum = 0;
					for( int i = 0; i < lentotal; fi++, di+=fio2.iFeaturesPerVector, i++ )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					fio2.pfVectors[(c*xx_dim*yy_dim + r*xx_dim + y)*fio2.iFeaturesPerVector + iFeature] = fSum;
				}
			}
			else
			{
				int c;
				for( c = 0; c < half_len; c++ )
				{
					int fi = half_len - c;
					int di = 0;
					float fSum = 0;
					for( ; fi < len; fi++, di++ )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					// Store in same plane, other image, switch rows/cols
					fio2.pfVectors[(c*xx_dim*yy_dim + r*xx_dim + y)*fio2.iFeaturesPerVector + iFeature] = fSum;
				}

				for( ; c < z_dim - half_len; c++ )
				{
					int fi = 0;
					int di = c - half_len;
					float fSum = 0;
					for( ; fi < len; fi++, di++ )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					// Store in same plane, other image, switch rows/cols
					fio2.pfVectors[(c*xx_dim*yy_dim + r*xx_dim + y)*fio2.iFeaturesPerVector + iFeature] = fSum;
				}

				for( ; c < z_dim; c++ )
				{
					int fi = 0;
					int di = c - half_len;
					float fSum = 0;
					len--;
					for( ; fi < len; fi++, di++ )
					{
						fSum += pfData[di]*pfFilter[fi];
					}
					// Store in same plane, other image, switch rows/cols
					fio2.pfVectors[(c*xx_dim*yy_dim + r*xx_dim + y)*fio2.iFeaturesPerVector + iFeature] = fSum;
				}
			}
		}
	}

	return 1;
}



int
blur_3d_simpleborders(
		FEATUREIO &fio1,
		FEATUREIO &fioTemp,
		FEATUREIO &fio2,
		int	iFeature,
		PpImage &ppImgFilter
		)
{
	int x_dim = fio1.x;
	int y_dim = fio1.y;
	int z_dim = fio1.z;

	int yy_dim = y_dim;
	int xx_dim = x_dim;

	int len = ppImgFilter.Cols();
	int half_len = ppImgFilter.Cols() / 2;
	assert( (len%2) == 1 );
//	assert( len <= fio1.x );
//	assert( len <= fio1.y );
//	assert( len <= fio1.z );

	float *pfFilter = (float*)ppImgFilter.ImageRow(0);

	// Allocate buffer
	int iBufferDim = fio1.x;
	if( fio1.y > iBufferDim )
	{
		iBufferDim = fio1.y;
	}
	if( fio1.z > iBufferDim )
	{
		iBufferDim = fio1.z;
	}
	iBufferDim += ppImgFilter.Cols();
	float *pfBufferIn = new float[iBufferDim];
	float *pfBufferOut = new float[iBufferDim];
	assert( pfBufferIn );
	assert( pfBufferOut );
	for( int c = 0; c < iBufferDim; c++ )
	{
		// Zero borders
		pfBufferIn[c] = 0;
	}


	// Blur x direction, store for y
	// fio1: y,x,z
	// fio2: x,y,z

	for( int z = 0; z < z_dim; z++ )
	{
		int iPlane = z*y_dim*x_dim;
		for( int r = 0; r < yy_dim; r++ )
		{
			float *pfData = fio1.pfVectors + iPlane + r*xx_dim;

			// Copy to buffer
			for( int c = 0; c < xx_dim; c++ )
			{
				pfBufferIn[c+half_len] = pfData[c];
			}

			// Filter
			filter_1d( pfFilter, pfBufferIn, pfBufferOut, len, iBufferDim );

			// Copy back to image
			for( int c = 0; c < xx_dim; c++ )
			{
				// Store in same plane, other image, switch rows/cols
				fio2.pfVectors[(iPlane+c*yy_dim+r)*fio2.iFeaturesPerVector + iFeature] = pfBufferOut[c+half_len];
			}
		}
	}

	// Blur y direction, store for z
	// fio2: x,y,z
	// fioTemp: z,x,y
	for( int c = 0; c < iBufferDim; c++ )
	{
		// Zero borders
		pfBufferIn[c] = 0;
	}
	for( int z = 0; z < z_dim; z++ )
	{
		int iPlane = z*xx_dim*yy_dim;

		for( int r = 0; r < xx_dim; r++ )
		{
			float *pfData = fio2.pfVectors + (iPlane + r*yy_dim)*fio2.iFeaturesPerVector;
			int len = 2*half_len+1;

			// Copy to buffer
			for( int c = 0; c < y_dim; c++ )
			{
				pfBufferIn[c+half_len] = pfData[c*fio2.iFeaturesPerVector ];
			}

			// Filter
			filter_1d( pfFilter, pfBufferIn, pfBufferOut, len, iBufferDim );

			// Copy back to image
			for( int c = 0; c < y_dim; c++ )
			{
				// Store in same plane, other image, switch rows/cols
				fioTemp.pfVectors[r*z_dim*yy_dim + c*z_dim + z] = pfBufferOut[c+half_len];
			}
		}
	}

	// Blur z direction, store as input
	// fioTemp: z,x,y
	// fio2: y,x,z
	for( int c = 0; c < iBufferDim; c++ )
	{
		// Zero borders
		pfBufferIn[c] = 0;
	}
	for( int y = 0; y < xx_dim; y++ )
	{
		int iPlane = y*z_dim*yy_dim;

		for( int r = 0; r < yy_dim; r++ )
		{
			float *pfData = fioTemp.pfVectors + iPlane + r*z_dim;
			int len = 2*half_len+1;

			// Copy to buffer
			for( int c = 0; c < z_dim; c++ )
			{
				pfBufferIn[c+half_len] = pfData[c];
			}

			// Filter
			filter_1d( pfFilter, pfBufferIn, pfBufferOut, len, iBufferDim );

			// Copy back to image
			for( int c = 0; c < z_dim; c++ )
			{
				// Store in same plane, other image, switch rows/cols
				fio2.pfVectors[(c*xx_dim*yy_dim + r*xx_dim + y)*fio2.iFeaturesPerVector + iFeature] = pfBufferOut[c+half_len];
			}
		}
	}
	//printf("%f\n", fio2.pfVectors[10651540]);
	delete [] pfBufferIn;
	delete [] pfBufferOut;

	return 1;
}

int
blur_3d_simpleborders_vyv(
		FEATUREIO &fio1,
		FEATUREIO &fioTemp,
		FEATUREIO &fio2,
		int iFeature,
		float fSigma,
		int iBorderSize
		)
{
	int x_dim = fio1.x;
	int y_dim = fio1.y;
	int z_dim = fio1.z;

	int yy_dim = y_dim;
	int xx_dim = x_dim;

	int len = iBorderSize;
	int half_len = iBorderSize / 2;
	assert( (len%2) == 1 );
//	assert( len <= fio1.x );
//	assert( len <= fio1.y );
//	assert( len <= fio1.z );

	// Allocate buffer
	int iBufferDim = fio1.x;
	if( fio1.y > iBufferDim )
	{
		iBufferDim = fio1.y;
	}
	if( fio1.z > iBufferDim )
	{
		iBufferDim = fio1.z;
	}
	iBufferDim += iBorderSize;
	float *pfBufferIn = new float[iBufferDim];
	float *pfBufferOut = new float[iBufferDim];
	//float *pfBufferExtra = new float[2*iBufferDim];
	assert( pfBufferIn );
	assert( pfBufferOut );
	//assert( pfBufferExtra );

	// Set up algorithm coefficients
	// VYV seems to work best, fastest & best quality for relatively small sigma (e.g. fSigma < 3 )
	// As for quality, the difference image for K=3 has noticable coherent blob structure compared to K=5
	// Repeatability tests are in order
	//  Timing: VYV: 35seconds(K=3) to 38seconds(K=5), FIR: 55seconds
	//  Features: VYV: 3800, FIR: 3500

	//vyv_coeffs vyv_c;
	//vyv_precomp(&vyv_c, fSigma, 5, 0.01);

	//deriche_coeffs deriche_c;
	//deriche_precomp(&deriche_c, fSigma, 2, 0.01);
	//sii_coeffs sii_c;
	//sii_precomp(&sii_c, fSigma, 3);
	//if( sii_buffer_size(sii_c,iBufferDim) > 2*iBufferDim )
	//{
	//	printf( "reallocating\n" );
	//	delete [] pfBufferExtra;
	//	pfBufferExtra = new float[sii_buffer_size(sii_c,iBufferDim)];
	//}

	// Blur x direction, store for y
	// fio1: y,x,z
	// fio2: x,y,z

	for( int z = 0; z < z_dim; z++ )
	{
		int iPlane = z*y_dim*x_dim;

		for( int c = 0; c < iBufferDim; c++ )
		{
			// Zero borders
			pfBufferIn[c] = 0;
		}

		for( int r = 0; r < yy_dim; r++ )
		{
			float *pfData = fio1.pfVectors + iPlane + r*xx_dim;

			// Copy to buffer
			for( int c = 0; c < xx_dim; c++ )
			{
				pfBufferIn[c+half_len] = pfData[c];
			}

			// Filter
			//filter_1d( pfFilter, pfBufferIn, pfBufferOut, len, iBufferDim );
			//vyv_gaussian_conv(vyv_c, pfBufferOut, pfBufferIn, xx_dim+len, 1);
			//box_gaussian_conv(pfBufferOut, pfBufferExtra, pfBufferIn, xx_dim+len, 1, fSigma, 3);
			//deriche_gaussian_conv(deriche_c,pfBufferOut, pfBufferExtra, pfBufferIn, xx_dim+len, 1);
			//sii_gaussian_conv(sii_c,pfBufferOut, pfBufferExtra, pfBufferIn, xx_dim+len, 1);

			// Copy back to image
			for( int c = 0; c < xx_dim; c++ )
			{
				// Store in same plane, other image, switch rows/cols
				fio2.pfVectors[(iPlane+c*yy_dim+r)*fio2.iFeaturesPerVector + iFeature] = pfBufferOut[c+half_len];
			}
		}
	}

	// Blur y direction, store for z
	// fio2: x,y,z
	// fioTemp: z,x,y
	for( int c = 0; c < iBufferDim; c++ )
	{
		// Zero borders
		pfBufferIn[c] = 0;
	}
	for( int z = 0; z < z_dim; z++ )
	{
		int iPlane = z*xx_dim*yy_dim;

		for( int c = 0; c < iBufferDim; c++ )
		{
			// Zero borders
			pfBufferIn[c] = 0;
		}

		for( int r = 0; r < xx_dim; r++ )
		{
			float *pfData = fio2.pfVectors + (iPlane + r*yy_dim)*fio2.iFeaturesPerVector;
			int len = 2*half_len+1;

			// Copy to buffer
			for( int c = 0; c < y_dim; c++ )
			{
				pfBufferIn[c+half_len] = pfData[c*fio2.iFeaturesPerVector ];
			}

			// Filter
			//filter_1d( pfFilter, pfBufferIn, pfBufferOut, len, iBufferDim );
			//vyv_gaussian_conv(vyv_c, pfBufferOut, pfBufferIn, y_dim+len, 1);
			//box_gaussian_conv(pfBufferOut, pfBufferExtra, pfBufferIn, y_dim+len, 1, fSigma, 3);
			//deriche_gaussian_conv(deriche_c,pfBufferOut, pfBufferExtra, pfBufferIn, y_dim+len, 1);
			//sii_gaussian_conv(sii_c,pfBufferOut, pfBufferExtra, pfBufferIn, y_dim+len, 1);

			// Copy back to image
			for( int c = 0; c < y_dim; c++ )
			{
				// Store in same plane, other image, switch rows/cols
				fioTemp.pfVectors[r*z_dim*yy_dim + c*z_dim + z] = pfBufferOut[c+half_len];
			}
		}
	}

	// Blur z direction, store as input
	// fioTemp: z,x,y
	// fio2: y,x,z
	for( int c = 0; c < iBufferDim; c++ )
	{
		// Zero borders
		pfBufferIn[c] = 0;
	}
	for( int y = 0; y < xx_dim; y++ )
	{
		int iPlane = y*z_dim*yy_dim;

		for( int c = 0; c < iBufferDim; c++ )
		{
			// Zero borders
			pfBufferIn[c] = 0;
		}

		for( int r = 0; r < yy_dim; r++ )
		{
			float *pfData = fioTemp.pfVectors + iPlane + r*z_dim;
			int len = 2*half_len+1;

			// Copy to buffer
			for( int c = 0; c < z_dim; c++ )
			{
				pfBufferIn[c+half_len] = pfData[c];
			}

			// Filter
			//filter_1d( pfFilter, pfBufferIn, pfBufferOut, len, iBufferDim );
			//vyv_gaussian_conv(vyv_c, pfBufferOut, pfBufferIn, z_dim+len, 1);
			//box_gaussian_conv(pfBufferOut, pfBufferExtra, pfBufferIn, z_dim+len, 1, fSigma, 3);
			//deriche_gaussian_conv(deriche_c,pfBufferOut, pfBufferExtra, pfBufferIn, z_dim+len, 1);
			//sii_gaussian_conv(sii_c,pfBufferOut, pfBufferExtra, pfBufferIn, z_dim+len, 1);

			// Copy back to image
			for( int c = 0; c < z_dim; c++ )
			{
				// Store in same plane, other image, switch rows/cols
				fio2.pfVectors[(c*xx_dim*yy_dim + r*xx_dim + y)*fio2.iFeaturesPerVector + iFeature] = pfBufferOut[c+half_len];
			}
		}
	}

	delete [] pfBufferIn;
	delete [] pfBufferOut;
	//delete [] pfBufferExtra;
	return 1;
}




//
// blur_2d()
//
// Performs a 2D blur on a 3D image where z =1. Assumes data is
// organized as y, x, z. Assumes black pixels at image border.
//
int
blur_2d(
		FEATUREIO &fio1,
		FEATUREIO &fioTemp,
		FEATUREIO &fio2,
		int	iFeature,
		PpImage &ppImgFilter
		)
{
	int x_dim = fio1.x;
	int y_dim = fio1.y;

	int yy_dim = y_dim;
	int xx_dim = x_dim;

	int len = ppImgFilter.Cols();
	int half_len = ppImgFilter.Cols() / 2;
	assert( (len%2) == 1 );
	assert( len <= fio1.x );
	assert( len <= fio1.y );

	float *pfFilter = (float*)ppImgFilter.ImageRow(0);

	// Blur y direction, store for x
	// fio1: y,x,z
	// fioTemp: x,y,z

	for( int r = 0; r < yy_dim; r++ )
	{
		float *pfData = fio1.pfVectors + r*xx_dim;
		int len = 2*half_len+1;

		int c;
		for( c = 0; c < half_len; c++ )
		{
			int fi = half_len - c;
			int di = 0;
			float fSum = 0;
			for( ; fi < len; fi++, di++ )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fioTemp.pfVectors[c*yy_dim+r] = fSum;
		}

		for( ; c < xx_dim - half_len; c++ )
		{
			int fi = 0;
			int di = c - half_len;
			float fSum = 0;
			for( ; fi < len; fi++, di++ )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fioTemp.pfVectors[c*yy_dim+r] = fSum;
		}

		for( ; c < xx_dim; c++ )
		{
			int fi = 0;
			int di = c - half_len;
			float fSum = 0;
			len--;
			for( ; fi < len; fi++, di++ )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fioTemp.pfVectors[c*yy_dim+r] = fSum;
		}
	}

	// Blur x direction, store for original
	// fioTemp: x,y,z
	// fio2: z,x,y

	for( int r = 0; r < xx_dim; r++ )
	{
		float *pfData = fioTemp.pfVectors + r*yy_dim;
		int len = 2*half_len+1;

		int c;
		for( c = 0; c < half_len; c++ )
		{
			int fi = half_len - c;
			int di = 0;
			float fSum = 0;
			for( ; fi < len; fi++, di++ )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fio2.pfVectors[(c*xx_dim+r)*fio2.iFeaturesPerVector + iFeature] = fSum;
		}

		for( ; c < yy_dim - half_len; c++ )
		{
			int fi = 0;
			int di = c - half_len;
			float fSum = 0;
			for( ; fi < len; fi++, di++ )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fio2.pfVectors[(c*xx_dim+r)*fio2.iFeaturesPerVector + iFeature] = fSum;
		}

		for( ; c < yy_dim; c++ )
		{
			int fi = 0;
			int di = c - half_len;
			float fSum = 0;
			len--;
			for( ; fi < len; fi++, di++ )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fio2.pfVectors[(c*xx_dim+r)*fio2.iFeaturesPerVector + iFeature] = fSum;
		}
	}

	return 1;
}

int
blur_2d_simpleborders(
		FEATUREIO &fio1,
		FEATUREIO &fioTemp,
		FEATUREIO &fio2,
		int	iFeature,
		PpImage &ppImgFilter
		)
{
	int x_dim = fio1.x;
	int y_dim = fio1.y;

	int yy_dim = y_dim;
	int xx_dim = x_dim;

	int len = ppImgFilter.Cols();
	int half_len = ppImgFilter.Cols() / 2;
	assert( (len%2) == 1 );
	//assert( len <= fio1.x );
	//assert( len <= fio1.y );

	float *pfFilter = (float*)ppImgFilter.ImageRow(0);

	// Allocate buffer
	int iBufferDim = fio1.x;
	if( fio1.y > fio1.x )
	{
		iBufferDim = fio1.y;
	}
	iBufferDim += ppImgFilter.Cols();
	float *pfBufferIn = new float[iBufferDim];
	float *pfBufferOut = new float[iBufferDim];
	assert( pfBufferIn );
	assert( pfBufferOut );
	for( int c = 0; c < iBufferDim; c++ )
	{
		// Zero borders
		pfBufferIn[c] = 0;
	}

	// Blur y direction, store for x
	// fio1: y,x,z
	// fioTemp: x,y,z

	for( int r = 0; r < yy_dim; r++ )
	{
		float *pfData = fio1.pfVectors + r*xx_dim;

		// Copy to buffer
		for( int c = 0; c < xx_dim; c++ )
		{
			pfBufferIn[c+half_len] = pfData[c];
		}

		// Filter
		filter_1d( pfFilter, pfBufferIn, pfBufferOut, len, iBufferDim );

		// Copy back to image
		for( int c = 0; c < xx_dim; c++ )
		{
			// Store in same plane, other image, switch rows/cols
			fioTemp.pfVectors[c*yy_dim+r] = pfBufferOut[c+half_len];
		}
	}

	// Blur x direction, store for original
	// fioTemp: x,y,z
	// fio2: z,x,y

	for( int r = 0; r < xx_dim; r++ )
	{
		float *pfData = fioTemp.pfVectors + r*yy_dim;

		// Copy to buffer
		for( int c = 0; c < yy_dim; c++ )
		{
			pfBufferIn[c+half_len] = pfData[c];
		}

		// Filter
		filter_1d( pfFilter, pfBufferIn, pfBufferOut, len, iBufferDim );

		// Copy back to image
		for( int c = 0; c < yy_dim; c++ )
		{
			// Store in same plane, other image, switch rows/cols
			fio2.pfVectors[(c*xx_dim+r)*fio2.iFeaturesPerVector + iFeature] = pfBufferOut[c+half_len];
		}
	}

	delete [] pfBufferIn;
	delete [] pfBufferOut;

	return 1;
}


//
// blur_2d()
//
// Performs a 2D blur on a 3D image where z =1. Assumes data is
// organized as y, x, z. Reflects image at border.
//
int
blur_2d_reflection(
		FEATUREIO &fio1,
		FEATUREIO &fioTemp,
		FEATUREIO &fio2,
		int	iFeature,
		PpImage &ppImgFilter
		)
{
	int x_dim = fio1.x;
	int y_dim = fio1.y;

	int yy_dim = y_dim;
	int xx_dim = x_dim;

	int len = ppImgFilter.Cols();
	int half_len = ppImgFilter.Cols() / 2;
	assert( (len%2) == 1 );
	assert( len < fio1.x );
	assert( len < fio1.y );

	float *pfFilter = (float*)ppImgFilter.ImageRow(0);

	// Blur y direction, store for x
	// fio1: y,x,z
	// fioTemp: x,y,z

	for( int r = 0; r < yy_dim; r++ )
	{
		float *pfData = fio1.pfVectors + r*xx_dim;
		int len = 2*half_len+1;

		int c;
		for( c = 0; c < half_len; c++ )
		{
			int fi = 0;
			int di = half_len - c;
			float fSum = 0;
			for( ; di > 0; fi++, di-- )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			for( ; fi < len; fi++, di++ )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fioTemp.pfVectors[c*yy_dim+r] = fSum;
		}

		for( ; c < xx_dim - half_len; c++ )
		{
			int fi = 0;
			int di = c - half_len;
			float fSum = 0;
			for( ; fi < len; fi++, di++ )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fioTemp.pfVectors[c*yy_dim+r] = fSum;
		}

		for( ; c < xx_dim; c++ )
		{
			int fi = 0;
			int di = c - half_len;
			float fSum = 0;
			for( ; di < xx_dim; fi++, di++ ) // Go to edge of image
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			di--;di--;	// Reverse di
			for( ; fi < len; fi++, di-- )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fioTemp.pfVectors[c*yy_dim+r] = fSum;
		}
	}

	// Blur x direction, store for original
	// fioTemp: x,y,z
	// fio2: z,x,y

	for( int r = 0; r < xx_dim; r++ )
	{
		float *pfData = fioTemp.pfVectors + r*yy_dim;
		int len = 2*half_len+1;

		int c;
		for( c = 0; c < half_len; c++ )
		{
			int fi = 0;
			int di = half_len - c;
			float fSum = 0;
			for( ; di > 0; fi++, di-- )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			for( ; fi < len; fi++, di++ )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fio2.pfVectors[(c*xx_dim+r)*fio2.iFeaturesPerVector + iFeature] = fSum;
		}

		for( ; c < yy_dim - half_len; c++ )
		{
			int fi = 0;
			int di = c - half_len;
			float fSum = 0;
			for( ; fi < len; fi++, di++ )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fio2.pfVectors[(c*xx_dim+r)*fio2.iFeaturesPerVector + iFeature] = fSum;
		}

		for( ; c < yy_dim; c++ )
		{
			int fi = 0;
			int di = c - half_len;
			float fSum = 0;
			for( ; di < yy_dim; fi++, di++ ) // Go to edge of image
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			di--;di--;	// Reverse di
			for( ; fi < len; fi++, di-- )
			{
				fSum += pfData[di]*pfFilter[fi];
			}
			// Store in same plane, other image, switch rows/cols
			fio2.pfVectors[(c*xx_dim+r)*fio2.iFeaturesPerVector + iFeature] = fSum;
		}
	}

	return 1;
}

int
gb3d_separable_2_blur3d_interleave(
		FEATUREIO &fioIn,		// Input
		FEATUREIO &fioTemp,		// Temporary
		FEATUREIO &fioOut1,		// Output
		FEATUREIO &fioOut2,		// Output
		int	iFeature,			// Index of interleaved feature in fio2
		float fSigma,			// Gaussian sigma parameter
		float fMinValue			// Min value
		)
{
	assert( fSigma > 0.0f );
	assert( fMinValue >= 0 && fMinValue < 1.0f );

	PpImage ppImgFilter;
	PpImage ppImgFilter1;
	PpImage ppImgFilter2;

	int iCols = calculate_gaussian_filter_size( fSigma, fMinValue );
	assert( (iCols % 2) == 1 );

	ppImgFilter.Initialize( 1, iCols, iCols*sizeof(float), sizeof(float)*8 );
	ppImgFilter1.Initialize( 1, iCols, iCols*sizeof(float), sizeof(float)*8 );
	ppImgFilter2.Initialize( 1, iCols, iCols*sizeof(float), sizeof(float)*8 );

	// Generate Gaussians
	int iDimension = fioIn.z > 1 ? 3 : 2;
	float fSigma1Div = pow( 2.0f, 2.0f/(float)iDimension );
	generate_gaussian_filter1d( ppImgFilter, fSigma, iCols/2 );
	generate_gaussian_filter1d( ppImgFilter1, fSigma/fSigma1Div, iCols/2 );

	// Set to maximum magnitude 1
	float fMax = ((float*)(ppImgFilter.ImageRow(0)))[iCols/2];
	float fMax1 = ((float*)(ppImgFilter1.ImageRow(0)))[iCols/2];
	for( int c = 0; c < ppImgFilter.Cols(); c++ )
	{
		((float*)(ppImgFilter.ImageRow(0)))[c] /= fMax;
		((float*)(ppImgFilter1.ImageRow(0)))[c] /= fMax1;
	}

	// Generate difference of Gaussian
	for( int c = 0; c < ppImgFilter.Cols(); c++ )
	{
		((float*)(ppImgFilter2.ImageRow(0)))[c] =
				((float*)(ppImgFilter.ImageRow(0)))[c] - ((float*)(ppImgFilter1.ImageRow(0)))[c];
	}

	// Normalize filters

	float fSum=0, fSum1=0, fSum2=0;
	for( int c = 0; c < ppImgFilter.Cols(); c++ )
	{
		fSum  += ((float*)(ppImgFilter.ImageRow(0)))[c];
		fSum1 += ((float*)(ppImgFilter1.ImageRow(0)))[c];
		fSum2 += ((float*)(ppImgFilter2.ImageRow(0)))[c];
	}
	for( int c = 0; c < ppImgFilter.Cols(); c++ )
	{
		((float*)(ppImgFilter.ImageRow(0)))[c]  /= fSum;
		((float*)(ppImgFilter1.ImageRow(0)))[c] /= fSum1;
		((float*)(ppImgFilter2.ImageRow(0)))[c] /= fSum2;
	}
	//fSum=0; fSum1=0; fSum2=0;
	//FILE *outfile = fopen( "filters.txt", "wt" );
	//for( int c = 0; c < ppImgFilter.Cols(); c++ )
	//{
	//	fSum  += ((float*)(ppImgFilter.ImageRow(0)))[c];
	//	fSum1 += ((float*)(ppImgFilter1.ImageRow(0)))[c];
	//	fSum2 += ((float*)(ppImgFilter2.ImageRow(0)))[c];
	//	fprintf( outfile, "%f\t%f\t%f\n",
	//		((float*)(ppImgFilter.ImageRow(0)))[c],
	//		((float*)(ppImgFilter1.ImageRow(0)))[c],
	//		((float*)(ppImgFilter2.ImageRow(0)))[c]
	//	);
	//}
	//fclose( outfile );

	if( fioIn.z == 1 )
	{
		//blur_2d_reflection( fioIn, fioTemp, fioOut1, iFeature, ppImgFilter1 );
		//blur_2d_reflection( fioIn, fioTemp, fioOut2, iFeature, ppImgFilter2 );
		blur_2d_simpleborders( fioIn, fioTemp, fioOut1, iFeature, ppImgFilter1 );
		blur_2d_simpleborders( fioIn, fioTemp, fioOut2, iFeature, ppImgFilter2 );
		return 1;
		//return blur_2d( fio1, fioTemp, fio2, iFeature, ppImgFilter );
	}
	else
	{
		return blur_3d( fioIn, fioTemp, fioOut1, iFeature, ppImgFilter1 );
		return blur_3d( fioIn, fioTemp, fioOut2, iFeature, ppImgFilter2 );
	}

	return 1;
}

int
gb3d_blur3d_interleave(
		FEATUREIO &fio1,		// Input
		FEATUREIO &fioTemp,		// Temporary
		FEATUREIO &fio2,		// Output
		int	iFeature,			// Index of interleaved feature in fio2
		float fSigma,			// Gaussian sigma parameter
		float fMinValue,
		int best_device_id			// Min value
		)
{

	assert( fSigma >= 0.0f );
	assert( fMinValue >= 0 && fMinValue < 1.0f );

	PpImage ppImgFilter;
	int iCols = calculate_gaussian_filter_size( fSigma, fMinValue );
	ppImgFilter.Initialize( 1, iCols, iCols*sizeof(float), sizeof(float)*8 );
	assert( (iCols % 2) == 1 );
	if( fSigma > 0.0f )
	{
		generate_gaussian_filter1d( ppImgFilter, fSigma, iCols/2 );
	}
	else
	{
		// Trivial length 1 filter
		assert( ppImgFilter.Cols() == 1 );
		float *pfValue = (float*)ppImgFilter.ImageRow(0);
		*pfValue = 1;
	}

	// Normalize filter

	float fSum =0;
	float *pfFilter = (float*)ppImgFilter.ImageRow(0);
	for( int c = 0; c < ppImgFilter.Cols(); c++ )
	{
		fSum += pfFilter[c];
	}
	for( int c = 0; c < ppImgFilter.Cols(); c++ )
	{
		pfFilter[c] /= fSum;
	}
	fSum =0;
	for( int c = 0; c < ppImgFilter.Cols(); c++ )
	{
		fSum += pfFilter[c];
	}

	if( fio1.z == 1 )
	{
		if( ppImgFilter.Cols() >= fio1.x || ppImgFilter.Cols() >= fio1.y )
		{
			// Fail
			//return 0;
		}
		return blur_2d_simpleborders( fio1, fioTemp, fio2, iFeature, ppImgFilter );
		//return blur_2d_reflection( fio1, fioTemp, fio2, iFeature, ppImgFilter );
		//return blur_2d( fio1, fioTemp, fio2, iFeature, ppImgFilter );
	}
	else
	{
		if( ppImgFilter.Cols() >= fio1.x || ppImgFilter.Cols() >= fio1.y || ppImgFilter.Cols() >= fio1.z )
		{
			// Fail
			//return 0;
		}
		/*if (ppImgFilter.Cols()>16) {
			return blur_3d_simpleborders( fio1, fioTemp, fio2, iFeature, ppImgFilter );
		}*//*
		auto start = high_resolution_clock::now();
		int val = blur_3d_simpleborders_CUDA_Row_Col_Shared_mem(fio1, fioTemp, fio2, iFeature, ppImgFilter);
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		if (ppImgFilter.Cols()==17) {
			cout << "\n" << duration.count() << " microseconds" << endl;
		}
		return val;*/
		//return blur_3d_simpleborders_CUDA_BLOCK_Shared_mem( fio1, fioTemp, fio2, iFeature, ppImgFilter );
		//return blur_3d_simpleborders_CUDA_Row_Rot_Shared_mem(fio1, fioTemp, fio2, iFeature, ppImgFilter);
		//if (fio1.scale <= 4) {
		if(best_device_id!=-1){
                        // model 6
			//return blur_3d_simpleborders_CUDA_3x1D_W_Rot_Shared_mem(fio1, fioTemp, fio2, iFeature, ppImgFilter, best_device_id);
			// model 5
                        return blur_3d_simpleborders_CUDA_Row_Col_Shared_mem(fio1, fioTemp, fio2, iFeature, ppImgFilter, best_device_id);
		}
		else{
			return blur_3d_simpleborders( fio1, fioTemp, fio2, iFeature, ppImgFilter );
		}
		//return blur_3d_simpleborders_CUDA_row_size(fio1, fioTemp, fio2, iFeature, ppImgFilter);
		//return blur_3d_simpleborders_CUDA_Shared_mem( fio1, fioTemp, fio2, iFeature, ppImgFilter );
		//return blur_3d_simpleborders_CUDA( fio1, fioTemp, fio2, iFeature, ppImgFilter );
		//return blur_3d_simpleborders( fio1, fioTemp, fio2, iFeature, ppImgFilter );
		//return blur_3d_simpleborders_vyv( fio1, fioTemp, fio2, iFeature, fSigma, ppImgFilter.Cols() );
		//return blur_3d( fio1, fioTemp, fio2, iFeature, ppImgFilter );
	}

	return 1;
}


int
gb3d_blur3d(
		FEATUREIO &fio1,		// Input
		FEATUREIO &fioTemp,		// Temporary
		FEATUREIO &fio2,		// Output
		float fSigma,			// Gaussian sigma parameter
		float fMinValue,			// Min value
		int best_device_id
		)
{
	return gb3d_blur3d_interleave( fio1, fioTemp, fio2, 0, fSigma, fMinValue , best_device_id);
}


int
gb3d_blur3d(
		FEATUREIO &fio1,
		FEATUREIO &fio2,
		float fSigma,			// Gaussian sigma parameter
		float fMinValue,			// Min value
		int best_device_id
		)
{
	return gb3d_blur3d( fio1, fio1, fio2, fSigma, fMinValue , best_device_id);
}
