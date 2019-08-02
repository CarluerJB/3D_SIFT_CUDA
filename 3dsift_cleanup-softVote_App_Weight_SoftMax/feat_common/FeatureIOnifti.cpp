
#include "FeatureIOnifti.h"
#include "nifti1_io.h"

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
// Read nifti, optionally convert to isotropic.
//
//int
//fioReadNifti(
//			 FEATUREIO &fioIn,
//			 char *pcFileName,
//			 int bIsotropic,
//			 int iRescale
//			 )
//{
//	/* Read the target and source images */
//	nifti_image *targetHeader = nifti_image_read(pcFileName,true);
//	if(targetHeader == NULL )
//	{
//		return -1;
//	}
//
//	if( targetHeader->data == NULL )
//	{
//		return -2;
//	}
//
//	reg_changeDatatype<float>(targetHeader);
//
//	FEATUREIO fioTmp;
//	memset( &fioTmp, 0, sizeof(FEATUREIO) );
//	memset( &fioIn, 0, sizeof(FEATUREIO) );
//
//	fioTmp.x = targetHeader->nx;
//	fioTmp.y = targetHeader->ny;
//	fioTmp.z = targetHeader->nz;
//	fioTmp.t = targetHeader->nt > 0 ? targetHeader->nt : 1;
//	fioTmp.iFeaturesPerVector = 1;
//	fioTmp.pfVectors = (float*)targetHeader->data;
//
//	if( bIsotropic &&
//		(targetHeader->dx != targetHeader->dy
//			|| targetHeader->dy != targetHeader->dz
//				|| targetHeader->dx != targetHeader->dz)
//				)
//	{
//		float fMinSize = targetHeader->dx;
//		if( targetHeader->dy < fMinSize )
//			fMinSize = targetHeader->dy;
//		if( targetHeader->dz < fMinSize )
//			fMinSize = targetHeader->dz;
//
//		fioIn.x = targetHeader->nx*targetHeader->dx / fMinSize;
//		fioIn.y = targetHeader->ny*targetHeader->dy / fMinSize;
//		fioIn.z = targetHeader->nz*targetHeader->dz / fMinSize;
//		fioIn.t = targetHeader->nt > 0 ? targetHeader->nt : 1;
//		fioIn.iFeaturesPerVector = 1;
//		if( !fioAllocate( fioIn ) )
//		{
//			// Not enough memory to resample :(
//			return -3;
//		}
//
//		float pfRescaleFactorsIJK[3];
//		pfRescaleFactorsIJK[0] = fMinSize / targetHeader->dx;
//		pfRescaleFactorsIJK[1] = fMinSize / targetHeader->dy;
//		pfRescaleFactorsIJK[2] = fMinSize / targetHeader->dz;
//
//		for( int z = 0; z < fioIn.z; z++ )
//		{
//			for( int y = 0; y < fioIn.y; y++ )
//			{
//				for( int x = 0; x < fioIn.x; x++ )
//				{
//					//float fPixel = fioGetPixelTrilinearInterp( fioTmp,
//					//		(x*fMinSize+0.5)/targetHeader->dx,
//					//		(y*fMinSize+0.5)/targetHeader->dy,
//					//		(z*fMinSize+0.5)/targetHeader->dz
//					//		);
//
//					// * This is the original code, I believe it is slightly
//					// incorrect, the 0.5 should be applied after multiplication/division
//					//float fPixel = fioGetPixelTrilinearInterp( fioTmp,
//					//		(x*fMinSize+0.5)/targetHeader->dx,
//					//		(y*fMinSize+0.5)/targetHeader->dy,
//					//		(z*fMinSize+0.5)/targetHeader->dz
//					//		);					
//
//					float fPixel = fioGetPixelTrilinearInterp( fioTmp,
//							(x*pfRescaleFactorsIJK[0]+0.5),
//							(y*pfRescaleFactorsIJK[1]+0.5),
//							(z*pfRescaleFactorsIJK[2]+0.5)
//							);
//
//					float *pfDestPix = fioGetVector( fioIn, x, y, z );
//					*pfDestPix = fPixel;
//				}
//			}
//		}
//
//		//fioWrite( fioIn, "out-iso"  );
//
//		nifti_image_free( targetHeader );
//	}
//	else
//	{
//		fioIn = fioTmp;
//	}
//
//	return 1;
//}
//



int
fioReadNifti(
			 FEATUREIO &fioIn,
			 char *pcFileName,
			 int bIsotropic,
			 int iRescale,
			 void *returnHead
			 )
{

	nifti_image *returnHeader = (nifti_image *)returnHead;
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

		// Determine which matrix to use - pick the non-zero one
		//float fSumQ = fabs(targetHeader->qto_ijk.m[0][0])+fabs(targetHeader->qto_ijk.m[0][2])+fabs(targetHeader->qto_ijk.m[0][2])+fabs(targetHeader->qto_ijk.m[0][3]);
		//float fSumS = fabs(targetHeader->sto_ijk.m[0][0])+fabs(targetHeader->sto_ijk.m[0][2])+fabs(targetHeader->sto_ijk.m[0][2])+fabs(targetHeader->sto_ijk.m[0][3]);
		//int bEqualMatrices = (memcmp( &(targetHeader->qto_ijk.m[0][0]), &(targetHeader->sto_ijk.m[0][0]), sizeof(targetHeader->qto_ijk.m) ) == 0);

		float x,y,z;

		// Generate multiplication factors for rescaling/upscaling
		// These factors shrink isotropic coordinates down to anisotropic.
		float pfRescaleFactorsIJK[3];
		pfRescaleFactorsIJK[0] = fMinSize / targetHeader->dx;
		pfRescaleFactorsIJK[1] = fMinSize / targetHeader->dy;
		pfRescaleFactorsIJK[2] = fMinSize / targetHeader->dz;

		//// This code is just to test the transforms
		//Feature3DInfo feat;
		//feat.x = 10; feat.y = 20; feat.z = 30;
		//feat.scale = 5;
		//feat.SimilarityTransform( &targetHeader->qto_xyz.m[0][0] );
		//feat.SimilarityTransform( &targetHeader->qto_ijk.m[0][0] );

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

		//feat.SimilarityTransform( &targetHeader->qto_xyz.m[0][0] );
		//feat.SimilarityTransform( &targetHeader->qto_ijk.m[0][0] );

		for( int z = 0; z < fioIn.z; z++ )
		{
			for( int y = 0; y < fioIn.y; y++ )
			{
				for( int x = 0; x < fioIn.x; x++ )
				{
					float pXYZ[3];

					// * This is the original code, I believe it is slightly
					// incorrect, the 0.5 should be applied after multiplication/division
					//float fPixel = fioGetPixelTrilinearInterp( fioTmp,
					//		(x*fMinSize+0.5)/targetHeader->dx,
					//		(y*fMinSize+0.5)/targetHeader->dy,
					//		(z*fMinSize+0.5)/targetHeader->dz
					//		);					

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

		//fioWrite( fioIn, "out-iso2"  );

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
fioWriteNifti(
			 FEATUREIO &fioIn,
			 void *targetHead,
			 char *pcFileName
			 )
{
	nifti_image *targetHeader = (nifti_image *)targetHead;

	char pcFileNameImg[400];
	char pcFileNameHdr[400];

	targetHeader->iname = pcFileNameImg;
	targetHeader->fname = pcFileNameHdr;
	sprintf( pcFileNameImg, "%s", pcFileName );
	sprintf( pcFileNameHdr, "%s", pcFileName );
	char *pch = strstr( pcFileNameImg, ".img" );
	if( pch )
	{
		pch = strstr( pcFileNameHdr, ".img" );
		pch[1] = 'h';pch[2] = 'd';pch[3] = 'r';
		//targetHeader->nifti_type = 2;
	}
	else if( pch = strstr( pcFileNameImg, ".hdr" ) )
	{
		pch = strstr( pcFileNameImg, ".hdr" );
		pch[1] = 'i';pch[2] = 'm';pch[3] = 'g';
		//targetHeader->nifti_type = 2;
	}
	else if( pch = strstr( pcFileNameImg, ".nii" ) )
	{
		targetHeader->iname = 0;
		//targetHeader->nifti_type = 1;
	}
	
	targetHeader->nifti_type = 0;

	nifti_image_write( targetHeader );
	//nifti_image_write_hdr_img( targetHeader, 1, "wb" );

	return 0;
}

