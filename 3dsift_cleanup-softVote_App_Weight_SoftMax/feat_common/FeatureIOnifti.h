

#ifndef __FEATUREIONIFTI_H__
#define __FEATUREIONIFTI_H__

#include "FeatureIO.h"

//
// fioReadNifti()
//
// Read nifty, optionally convert to isotropic.
//
//int
//fioReadNifti(
//			 FEATUREIO &fioIn,
//			 char *pcFileName,
//			 int bIsotropic = 0,
//			 int iRescale = 0
//			 );
//


int
fioReadNifti(
			 FEATUREIO &fioIn,
			 char *pcFileName,
			 int bIsotropic = 0,
			 int iRescale = 0,
			 void * returnHead = 0 //nifti_image *returnHeader = 0
			 );

int
fioWriteNifti(
			 FEATUREIO &fioIn,
			 void *targetHead,// nifti_image *targetHeader,
			 char *pcFileName
			 );

#endif
