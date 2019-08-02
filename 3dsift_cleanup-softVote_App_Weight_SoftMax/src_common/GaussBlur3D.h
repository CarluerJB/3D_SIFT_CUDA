
#ifndef __GAUSSBLUR3D_H__
#define __GAUSSBLUR3D_H__

#include "FeatureIO.h"

//
// blur_3d()
//
// Performs a 3D blur on an image. Assumes data is organized
// as y, x, z. Assumes black pixels at image border.
//
int
gb3d_blur3d(
		FEATUREIO &fio1,		// Input, destroyed :(
		FEATUREIO &fio2,		// Output
		float fSigma,			// Gaussian sigma parameter
		float fMinValue,			// Min value
		int best_device_id
		);

int
gb3d_blur3d(
		FEATUREIO &fio1,		// Input
		FEATUREIO &fioTemp,		// Temporary
		FEATUREIO &fio2,		// Output
		float fSigma,			// Gaussian sigma parameter
		float fMinValue,			// Min value
		int best_device_id
		);

//
// gb3d_blur3d_interleave()
//
// Perimts interleaving of features.
//
int
gb3d_blur3d_interleave(
		FEATUREIO &fio1,		// Input
		FEATUREIO &fioTemp,		// Temporary
		FEATUREIO &fio2,		// Output
		int	iFeature,			// Index of interleaved feature in fio2
		float fSigma,			// Gaussian sigma parameter
		float fMinValue,			// Min value
		int best_device_id
		);

//
// gb3d_separable_2_blur3d_interleave()
//
// Creates creates a binary separable filter
// which is equivalent to the Gaussian specified.
//
int
gb3d_separable_2_blur3d_interleave(
		FEATUREIO &fioIn,		// Input
		FEATUREIO &fioTemp,		// Temporary
		FEATUREIO &fioOut1,		// Output
		FEATUREIO &fioOut2,		// Output
		int	iFeature,			// Index of interleaved feature in fio2
		float fSigma,			// Gaussian sigma parameter
		float fMinValue			// Min value
		);

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
		  );

#endif
