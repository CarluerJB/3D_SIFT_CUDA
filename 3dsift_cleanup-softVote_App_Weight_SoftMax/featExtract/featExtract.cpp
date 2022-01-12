#include <stdio.h>
#include <assert.h>

#include "nifti1_io.h"
#include "MultiScale.h"
#include "FeatureIO.h"
#include "PpImage.h"
#include "SIFT_cuda_Tools.cuh"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

template <class NewTYPE, class DTYPE>
void reg_changeDatatype1(nifti_image *image)
{
	// the initial array is saved and freeed
	DTYPE *initialValue = (DTYPE *)malloc(image->nvox*sizeof(DTYPE));
	memcpy(initialValue, image->data, image->nvox*sizeof(DTYPE));

	// the new array is allocated and then filled
	if(sizeof(NewTYPE)==sizeof(unsigned char)) image->datatype = NIFTI_TYPE_UINT8;
	else if(sizeof(NewTYPE)==sizeof(float)) image->datatype = NIFTI_TYPE_FLOAT32;
	else if(sizeof(NewTYPE)==sizeof(double)) image->datatype = NIFTI_TYPE_FLOAT64;
	else{
		printf("err\treg_changeDatatype\tOnly change to unsigned char, float or double are supported\n");
		free(initialValue);
		return;
	}
	free(image->data);
	image->nbyper = sizeof(NewTYPE);
	image->data = (void *)calloc(image->nvox,sizeof(NewTYPE));
	NewTYPE *dataPtr = static_cast<NewTYPE *>(image->data);
	for(unsigned int i=0; i<image->nvox; i++)
		dataPtr[i] = (NewTYPE)(initialValue[i]);

	free(initialValue);
	return;
}

template <class NewTYPE>
void reg_changeDatatype(nifti_image *image)
{
	switch(image->datatype){
		case NIFTI_TYPE_UINT8:
			reg_changeDatatype1<NewTYPE,unsigned char>(image);
			break;
		case NIFTI_TYPE_INT8:
			reg_changeDatatype1<NewTYPE,char>(image);
			break;
		case NIFTI_TYPE_UINT16:
			reg_changeDatatype1<NewTYPE,unsigned short>(image);
			break;
		case NIFTI_TYPE_INT16:
			reg_changeDatatype1<NewTYPE,short>(image);
			break;
		case NIFTI_TYPE_UINT32:
			reg_changeDatatype1<NewTYPE,unsigned int>(image);
			break;
		case NIFTI_TYPE_INT32:
			reg_changeDatatype1<NewTYPE,int>(image);
			break;
		case NIFTI_TYPE_FLOAT32:
			reg_changeDatatype1<NewTYPE,float>(image);
			break;
		case NIFTI_TYPE_FLOAT64:
			reg_changeDatatype1<NewTYPE,double>(image);
			break;
		default:
			printf("err\treg_changeDatatype\tThe initial image data type is not supported\n");
			return;
	}
}

//
// fioReadNifti()
//
// Read nifty, optionally convert to isotropic.
//
int
fioReadNifti(
			 FEATUREIO &fioIn,
			 char *pcFileName,
			 int bIsotropic = 0,
			 int iRescale = 0,
			 nifti_image *returnHeader = 0
			 )
{
	/* Read the target and source images */
	nifti_image *targetHeader = nifti_image_read(pcFileName,true);
	if(targetHeader == NULL )
	{
		return -1;
	}

	if( targetHeader->data == NULL )
	{
		return -2;
	}

	reg_changeDatatype<float>(targetHeader);

	FEATUREIO fioTmp;
	memset( &fioTmp, 0, sizeof(FEATUREIO) );
	memset( &fioIn, 0, sizeof(FEATUREIO) );

	fioTmp.x = targetHeader->nx;
	fioTmp.y = targetHeader->ny;
	fioTmp.z = targetHeader->nz;
	fioTmp.t = targetHeader->nt > 0 ? targetHeader->nt : 1;
	fioTmp.iFeaturesPerVector = 1;
	fioTmp.pfVectors = (float*)targetHeader->data;

	if( bIsotropic &&
		(targetHeader->dx != targetHeader->dy
			|| targetHeader->dy != targetHeader->dz
				|| targetHeader->dx != targetHeader->dz)
				)
	{
		float fMinSize = targetHeader->dx;
		if( targetHeader->dy < fMinSize )
			fMinSize = targetHeader->dy;
		if( targetHeader->dz < fMinSize )
			fMinSize = targetHeader->dz;

		fioIn.x = targetHeader->nx*targetHeader->dx / fMinSize;
		fioIn.y = targetHeader->ny*targetHeader->dy / fMinSize;
		fioIn.z = targetHeader->nz*targetHeader->dz / fMinSize;
		fioIn.t = targetHeader->nt > 0 ? targetHeader->nt : 1;
		fioIn.iFeaturesPerVector = 1;
		if( !fioAllocate( fioIn ) )
		{
			// Not enough memory to resample :(
			return -3;
		}

		float x,y,z;

		// Generate multiplication factors for rescaling/upscaling
		// These factors shrink isotropic coordinates down to anisotropic.
		float pfRescaleFactorsIJK[3];
		pfRescaleFactorsIJK[0] = fMinSize / targetHeader->dx;
		pfRescaleFactorsIJK[1] = fMinSize / targetHeader->dy;
		pfRescaleFactorsIJK[2] = fMinSize / targetHeader->dz;

		// This code is just to test the transforms
		Feature3DInfo feat;
		feat.x = 10; feat.y = 20; feat.z = 30;
		feat.scale = 5;
		feat.SimilarityTransform( &targetHeader->qto_xyz.m[0][0] );
		feat.SimilarityTransform( &targetHeader->qto_ijk.m[0][0] );

		// Apply rescale factors ijk, one per factor per column.
		// These will be used to compute feature coordinates in xyz
		// Note: scaling factors operate on ijk voxel coordinates, so
		// apply one rescaling/normalization factor per transform matrix column.
		// Essentially, we rescale the direction cosine angles.
		for( int i = 0; i < 3; i++ )
		{
			for( int j = 0; j < 3; j++ )
			{
				targetHeader->qto_xyz.m[i][j] *= pfRescaleFactorsIJK[j];

				if( targetHeader->sform_code > 0 )
					targetHeader->sto_xyz.m[i][j] *= pfRescaleFactorsIJK[j];
			}
		}

		// Compute inverses
		targetHeader->qto_ijk = nifti_mat44_inverse( targetHeader->qto_xyz );
		if( targetHeader->sform_code > 0 )
			targetHeader->sto_ijk = nifti_mat44_inverse( targetHeader->sto_xyz );

		feat.SimilarityTransform( &targetHeader->qto_xyz.m[0][0] );
		feat.SimilarityTransform( &targetHeader->qto_ijk.m[0][0] );

		for( int z = 0; z < fioIn.z; z++ )
		{
			for( int y = 0; y < fioIn.y; y++ )
			{
				for( int x = 0; x < fioIn.x; x++ )
				{
					float pXYZ[3];
					float fPixel = fioGetPixelTrilinearInterp( fioTmp,
							(x*pfRescaleFactorsIJK[0]+0.5),
							(y*pfRescaleFactorsIJK[1]+0.5),
							(z*pfRescaleFactorsIJK[2]+0.5)
							);

					float *pfDestPix = fioGetVector( fioIn, x, y, z );
					*pfDestPix = fPixel;
				}
			}
		}

		// Reset changed fields
		targetHeader->dx = fMinSize;
		targetHeader->dy = fMinSize;
		targetHeader->dz = fMinSize;
	}
	else
	{
		// Allocate & copy image to new data structure
		// (old will be deleted ... )
		fioIn = fioTmp;
		fioAllocate( fioIn );
		fioCopy( fioIn, fioTmp );
	}

	// Save return target header information
	if( returnHeader != 0 )
		memcpy( returnHeader, targetHeader, sizeof(nifti_image) );

	nifti_image_free( targetHeader );
	return 1;
}

int
print_options(
			  )
{
		printf( "Volumetric local feature extraction v1.1\n");
		printf( "Usage: %s [options] <input image> <output features>\n", "featExtract" );
		printf( "  <input image>: nifti (.nii,.hdr,.nii.gz).\n" );// or raw input volume (IEEE 32-bit float, little endian).\n" );
		printf( "  <output features>: output file with features.\n" );
		printf( " [options]\n" );
		printf( "  -w         : output feature geometry in world coordinates, NIFTI qto_xyz matrix (default is voxel units).\n" );
		printf( "  -2+        : double input image size.\n" );
		printf( "  -2-        : halve input image size.\n" );
		printf( "  -d[1-9]    : set device id to be used.\n" );
		return 0;
}

int
check_best_device(){
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	int best_device_id = 0;
	int best_cc_major = 0;
	int best_cc_minor = 0;
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		if (prop.major>=best_cc_major) {
			if(prop.major>best_cc_major){
				best_cc_major = prop.major;
				best_cc_minor = prop.minor;
				best_device_id=i+1;
			}
			else{
				if (prop.minor=best_cc_minor) {
					best_cc_major = prop.major;
					best_cc_minor = prop.minor;
					best_device_id=i+1;
				}
			}
		}
	}
	return best_device_id;
}

int get_num_device(){
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	return nDevices;
}


int
main(
	 int argc,
	 char **argv
)
{
	// Write the code to test the scale selection stuff

	FEATUREIO	fioIn;
	FEATUREIO	*d_fioIn;
	PpImage		ppImgIn;

	if( argc < 3 )
	{
		print_options();
		return -1;
	}
	int best_device_id = -1;
	int iArg = 1;
	int bDoubleImageSize = 0;
	int bOutputText = 1;
	int bWorldCoordinates = 0;
	int bIsotropicProcessing = 0;
	int iFlipCoord = 0; // Flip a coordinate system: x=1,y=2,z=3
	float fEigThres = 140;
	int bMultiModal = 0;
	while( argv[iArg][0] == '-' )
	{
		switch( argv[iArg][1] )
		{
			// Double initial image resolution option
		case '2':
			// Option to double initial image size, get 8 times as many features...
			bDoubleImageSize = 1;
			if( argv[iArg][2] == '-' )
			{
				// Halve image resolution
				bDoubleImageSize = -1;
			}
			iArg++;
			break;

			case 'd':
				// Option to use GPU...
				if( argv[iArg][2] - '0' < 0 || argv[iArg][2] - '0' > get_num_device())
				{
					printf( "Error: unknown device: %d\n", argv[iArg][2] - '0' );
					print_options();
					return -1;
					break;
				}
				else{
					best_device_id = argv[iArg][2] - '0';
				}
				iArg++;
				break;

		case 'w': case 'W':
			// Option to output world coordinates
			// Performs isotropic extraction, otherwise world coordinates
			// doesn't make sense
			bWorldCoordinates = 1;
			bIsotropicProcessing = 1;
			if( argv[iArg][2] == 's' || argv[iArg][2] == 'S' )
			{
				// Optionally use the nifti sform coordinate system, if available
				bWorldCoordinates = 2;
			}
			iArg++;
			break;

		default:
			printf( "Error: unknown command line argument: %s\n", argv[iArg] );
			print_options();
			return -1;
			break;
		}
	}

	if (argc - iArg < 2) {
		print_options();
		return -1;
	}
	
	printf( "Extracting features: %s\n", argv[iArg] );

	nifti_image returnHeader;
	fioIn.device=best_device_id;
	if( fioReadNifti( fioIn, argv[iArg], bIsotropicProcessing, 0, &returnHeader ) < 0 )
	{
		printf( "Error: could not read input file: %s\n", argv[iArg] );
		return -1;
	}

	// Define initial image pixel size
	float fInitialBlurScale = 1.0f;
	if( bDoubleImageSize != 0 )
	{
		if( bDoubleImageSize == 1 )
		{
			// Performing image doubling halves the size of pixels
			fioDoubleSize( fioIn );
			fInitialBlurScale *= 0.5;
		}
		else if( bDoubleImageSize == -1 )
		{
			// Reduce image size, initial pi
			FEATUREIO fioTmp = fioIn;
			fioIn.x /= 2;
			fioIn.y /= 2;
			fioIn.z /= 2;
			fioAllocate( fioIn );
			fioSubSample2DCenterPixel( fioTmp, fioIn );
			fioDelete( fioTmp );
		}
	}

	//
	// Simple extraction & visualization
	// For binary distribution

	if( fioIn.z <= 1 )
	{
 		printf( "Could not read volume: %s\n", argv[iArg] );
		return -1;
	}

	printf( "Input image: i=%d j=%d k=%d\n", fioIn.x, fioIn.y, fioIn.z );

	vector<Feature3D> vecFeats3D;
	int iReturn;
	try
	{
		// to fix here!!
		// Something is broken here, extraction returns way too many features. Release version might work.
		//iReturn = msGeneratePyramidDOG3D( fioIn, vecFeats3D, fInitialBlurScale, 0, 0, fEigThres );
		iReturn = msGeneratePyramidDOG3D_efficient( fioIn, vecFeats3D, best_device_id, fInitialBlurScale, 0, 0, fEigThres );
	}
	catch (...)
	{
		printf( "Error: could not extract features, insufficient memory.\n" );
		return -1;
	}
	if( iReturn != 1 )
	{
		printf( "Error: could not extract features, insufficient memory.\n" );
		return -1;
	}

	// Prepare scaling factor
	float fSizeFactor = 1;
	if( bDoubleImageSize > 0 )
		fSizeFactor /= 2;
	else if( bDoubleImageSize < 0 )
		fSizeFactor *= 2;

	// Prepare transform to world coordinates
	// Scale should be isotropic in matrix gto_xyz
	float pfScale[3];
	float fScaleSum = 0;
	float pfRot3x3[3][3];
	mat44 m_current;

	if( bWorldCoordinates )
	{
		if( !bIsotropicProcessing )
		{
			printf( "Error: world coordinate output with anisotropic extraction.\n" );
			return -1;
		}
		// Feature orientation vectors need to be rotated to xyz
		// Points need transformed
		// Scale needs to be adjusted

		mat44 *pm = &returnHeader.qto_xyz;
		if( bWorldCoordinates == 2 )
		{
			if( returnHeader.sform_code > 0 )
			{
				pm = &returnHeader.sto_xyz;
			}
			else
			{
				printf( "Error: sform_code <= 0, output to qto_xyz instead of sto_xyz" );
			}
		}

		memcpy( &m_current.m, pm->m, sizeof(mat44) );

		for( int i = 0; i < 3; i++ )
		{
			// Figure out scaling - should be isotropic
			pfScale[i] = vec3D_mag( &pm->m[i][0] );
			fScaleSum += pfScale[i];

			// Create rotation matrix - should be able to rescale
			memcpy( &pfRot3x3[i][0], &pm->m[i][0], 3*sizeof(float) );
			vec3D_norm_3d( &pfRot3x3[i][0] );
		}
		fScaleSum /= 3;
	}
	int brief=0;
	vector<LOCATION_VALUE_XYZ> RandomIndexX, RandomIndexY;
	msGenerateBRIEFindex(RandomIndexX, RandomIndexY, 64, fioIn);
	for( int iFeat = 0; iFeat < vecFeats3D.size(); iFeat++ )
	{
		Feature3DInfo &featHere = vecFeats3D[iFeat];
		vecFeats3D[iFeat].NormalizeData();

		if (brief) {
			msResampleFeaturesBRIEF(vecFeats3D[iFeat], RandomIndexX, RandomIndexY);
			/*int direction[64];
			for(int h=0; h<64; h++){
				if (vecFeats3D[iFeat].m_pfPC[h]<0) {
					direction[h]=-1;
				}
				else{direction[h]=1;}
			}*/
			vecFeats3D[iFeat].NormalizeDataRankedPCs();
			/*for(int h=0; h<64; h++){
				vecFeats3D[iFeat].m_pfPC[h]*=direction[h];
			}*/
		}
		else{
			msResampleFeaturesGradientOrientationHistogram(vecFeats3D[iFeat]);
			vecFeats3D[iFeat].NormalizeDataRankedPCs();
		}

		// Incorporate scale/size factor on pixel ijk coordinates
		vecFeats3D[iFeat].x *= fSizeFactor;
		vecFeats3D[iFeat].y *= fSizeFactor;
		vecFeats3D[iFeat].z *= fSizeFactor;
		vecFeats3D[iFeat].scale *= fSizeFactor;

		if( bWorldCoordinates )
		{
			if( !bIsotropicProcessing )
			{
				printf( "Error: world coordinate output with anisotropic extraction.\n" );
				return -1;
			}

			// Convert feature coordinates
			float pfIJK[4];
			float pfXYZ[4];
			pfIJK[0] = vecFeats3D[iFeat].x;
			pfIJK[1] = vecFeats3D[iFeat].y;
			pfIJK[2] = vecFeats3D[iFeat].z;
			pfIJK[3] = 1;
			mult_4x4_vector(&m_current.m[0][0], pfIJK, pfXYZ );
			vecFeats3D[iFeat].x = pfXYZ[0];
			vecFeats3D[iFeat].y = pfXYZ[1];
			vecFeats3D[iFeat].z = pfXYZ[2];

			// Convert scale
			vecFeats3D[iFeat].scale *= fScaleSum;

			// Rotate orientation vectors
			// Feature ori matrix contains direction cosine vectors in columns.
			// Rotation should multiply ori from the left hand side:
			float pfOri[3][3];
			float pfOriOut[3][3];
			invert_3x3<float,float>( vecFeats3D[iFeat].ori, pfOri );
			mult_3x3_matrix<float,float>( pfRot3x3, pfOri, pfOriOut );
			invert_3x3<float,float>( pfOriOut, vecFeats3D[iFeat].ori );
		}
	}

	// Now convert
	char *ppcComments[3];
	char pcComment1[200];
	char pcComment2[200];
	char pcComment3[400];
	sprintf( pcComment1, "Extraction Voxel Resolution (ijk) : %d %d %d", fioIn.x, fioIn.y, fioIn.z );
	sprintf( pcComment2, "Extraction Voxel Size (mm)  (ijk) : %f %f %f", 1.0f*returnHeader.dx, 1.0f*returnHeader.dy, 1.0f*returnHeader.dz );
	if( bWorldCoordinates )
	{
		if( bWorldCoordinates == 1 )
		{
			sprintf( pcComment3, "Feature Coordinate Space: millimeters (qto_xyz) : %f %f %f %f %f %f %f %f %f %f %f %f 0.0 0.0 0.0 1.0",
				1.0f*m_current.m[0][0],1.0f*m_current.m[0][1],1.0f*m_current.m[0][2],1.0f*m_current.m[0][3],
				1.0f*m_current.m[1][0],1.0f*m_current.m[1][1],1.0f*m_current.m[1][2],1.0f*m_current.m[1][3],
				1.0f*m_current.m[2][0],1.0f*m_current.m[2][1],1.0f*m_current.m[2][2],1.0f*m_current.m[2][3] );
		}
		else if( bWorldCoordinates == 2 )
		{
			sprintf( pcComment3, "Feature Coordinate Space: millimeters (sto_xyz) : %f %f %f %f %f %f %f %f %f %f %f %f 0.0 0.0 0.0 1.0",
				1.0f*m_current.m[0][0],1.0f*m_current.m[0][1],1.0f*m_current.m[0][2],1.0f*m_current.m[0][3],
				1.0f*m_current.m[1][0],1.0f*m_current.m[1][1],1.0f*m_current.m[1][2],1.0f*m_current.m[1][3],
				1.0f*m_current.m[2][0],1.0f*m_current.m[2][1],1.0f*m_current.m[2][2],1.0f*m_current.m[2][3] );
		}
	}
	else
	{
		sprintf( pcComment3, "Feature Coordinate Space: voxels: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0" );
	}

	ppcComments[0] = pcComment1;
	ppcComments[1] = pcComment2;
	ppcComments[2] = pcComment3;
	if( bOutputText )
	{
		msFeature3DVectorOutputText( vecFeats3D, argv[iArg+1], fEigThres, 3, ppcComments );
	}
	else
	{
		msFeature3DVectorOutputBin( vecFeats3D, argv[iArg+1], fEigThres );
	}

	printf( "\nDone.\n" );

	return 0;
}
