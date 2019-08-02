

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "FeatureIO.h"
#include "PpImage.h"
#include "PpImageFloatOutput.h"
#include "svdcmp.h"
#include "SIFT_cuda_Tools.cuh"

#include <cuda.h>
#include <cuda_runtime.h>


#include <vector>
#include <algorithm>
using namespace std;

#define PI 3.1415926535897932384626433832795

static int
find_min_max(
			 float *pfFeatures,
			 const int &iCount,
			 float &fMin,			// Min value
			 float &fMax				// Max value
			 )
{
	fMin = pfFeatures[0];
	fMax = pfFeatures[0];
	for( int i = 0; i < iCount; i++ )
	{
		if( pfFeatures[i] < fMin )
		{
			fMin = pfFeatures[i];
		}
		if( pfFeatures[i] > fMax )
		{
			fMax = pfFeatures[i];
		}
	}

	return 1;
}


//
// read_info_file()
//
// Read header, return fields successfully identified.
//
static int
read_info_file(
			   FEATUREIO &fio,
			   char *pcFileName,
			   char *pcDataFileName = 0 // Name of the data file associated with this info fil
				)
{
	FILE *infile = fopen( pcFileName, "rt" );
	if( !infile )
	{
		return 0;
	}

	char pcLine[300];

	int bGotF=0,bGotX=0,bGotY=0,bGotZ=0,bGotD=0;
	fio.t = 1;

	// Set default data file name
	if( pcDataFileName )
	{
		//sprintf( pcDataFileName, "%s", pcFileName );
	}

	while( fgets( pcLine, sizeof(pcLine), infile ) )
	{
		if( strchr( pcLine, ':' ) )
		{
			char *pch = &pcLine[0];
			while( *pch == ' ' || *pch == '\t' )
			{
				// Eat whitespace
				pch++;
			}
			switch( pch[0] )
			{
			case 'f':
			case 'F':
				bGotF=1;
				sscanf( pch, "Features: %d\n", &fio.iFeaturesPerVector );
				break;

			case 'x':
			case 'X':
				bGotX=1;
				sscanf( pch, "x: %d\n", &fio.x );
				break;

			case 'y':
			case 'Y':
				bGotY=1;
				sscanf( pch, "y: %d\n", &fio.y );
				break;

			case 'z':
			case 'Z':
				bGotZ=1;
				sscanf( pch, "z: %d\n", &fio.z );
				break;

			case 't':
			case 'T':
				sscanf( pch, "t: %d\n", &fio.t );
				break;

			case 'd':
			case 'D':
				bGotD=1;
				if( pcDataFileName )
				{
					if( sscanf( pch, "data: %s\n", pcDataFileName ) != 1 )
					{
						sscanf( pch, "Data: %s\n", pcDataFileName );
					}
				}
				break;
			}
		}
	}

	fclose( infile );

	if( !bGotD && pcDataFileName )
	{
		sprintf( pcDataFileName, "%s", pcFileName );
		char *pch = strchr( pcDataFileName, '.' );
		if( pch )
		{
			if( strstr( pch, ".txt" ) )
			{
				pch[1] = 'b';
				pch[2] = 'i';
				pch[3] = 'n';
				pch[4] = '\0';
			}
		}
	}

	int iReturn = 0;

	iReturn = bGotF | (iReturn<<1);
	iReturn = bGotX | (iReturn<<1);
	iReturn = bGotY | (iReturn<<1);
	iReturn = bGotZ | (iReturn<<1);
	iReturn = bGotD | (iReturn<<1);

	int bGotAllDimentions = bGotF&&bGotX&&bGotY&&bGotZ;
	//if( !bGotAllDimentions )
	//{
	//	return 0;
	//}

	return iReturn;
}

static int
read_info_file_data(
			   FEATUREIO &fio,
			   char *pcFileName
				)
{
	FILE *infile = fopen( pcFileName, "rt" );
	if( !infile )
	{
		return 0;
	}

	int i;

	char *pchTemp;
	char pcLine[300];

	while( fgets( pcLine, sizeof(pcLine), infile ) )
	{
		if( strchr( pcLine, ':' ) )
		{
			switch( pcLine[0] )
			{

			case 'm':	// Read feature means
			case 'M':

				pchTemp = strchr( pcLine, ':' ) + 1;
				pchTemp = strtok( pchTemp, " \t" );

				for( int i = 0; i < fio.iFeaturesPerVector; i++ )
				{
					sscanf( pchTemp, "%f", &fio.pfMeans[i] );
					pchTemp = strtok( 0, " \t" );
				}
				break;

			case 'v':	// Read feature variances
			case 'V':

				pchTemp = strchr( pcLine, ':' ) + 1;
				pchTemp = strtok( pchTemp, " \t" );

				for( int i = 0; i < fio.iFeaturesPerVector; i++ )
				{
					sscanf( pchTemp, "%f", &fio.pfVarrs[i] );
					pchTemp = strtok( 0, " \t" );
				}
				break;
			}
		}
	}


	fclose( infile );

	return 1;
}

static int
write_info_file(
			   FEATUREIO &fio,
			   char *pcFileName
				)
{
	FILE *outfile = fopen( pcFileName, "wt" );
	if( !outfile )
	{
		return 0;
	}

	int i;

	fprintf( outfile, "Features:\t%d\n", fio.iFeaturesPerVector );
	fprintf( outfile, "x:\t%d\n", fio.x );
	fprintf( outfile, "y:\t%d\n", fio.y );
	fprintf( outfile, "z:\t%d\n", fio.z );
	fprintf( outfile, "t:\t%d\n", fio.t );

	char pcDataFileName[400];
	sprintf( pcDataFileName, "%s", pcFileName );
	char *pch = strrchr( pcDataFileName, '.' );
	// pcFileName should always finish with .txt, so pch should never be null
	assert( pch );
	if( pch )
	{
		pch[1] = 'i';//'b';
		pch[2] = 'm';//'i';
		pch[3] = 'g';//'n';
		pch[4] = 0;	 //0;

		// Remove all directory info
		pch = strrchr( pcDataFileName, '\\' );
		if( pch )
		{
			// Windows directory found
			pch++;
		}
		else
		{
			pch = strrchr( pcDataFileName, '/' );
			if( pch )
			{
				// Linux directory found
				pch++;
			}
			else
			{
				// No directory, go with entire name
				pch = &(pcDataFileName[0]);
			}
		}
		fprintf( outfile, "data:\t%s\n", pch );
	}

	if( fio.pfMeans != 0 )
	{
		fprintf( outfile, "means:" );
		for( int i = 0; i < fio.iFeaturesPerVector; i++ )
		{
			fprintf( outfile, " %f", fio.pfMeans[i] );
		}
		fprintf( outfile, "\n" );
	}

	if( fio.pfVarrs != 0 )
	{
		fprintf( outfile, "variances:" );
		for( int i = 0; i < fio.iFeaturesPerVector; i++ )
		{
			fprintf( outfile, " %f", fio.pfVarrs[i] );
		}
		fprintf( outfile, "\n" );
	}

	fclose( outfile);

	return 1;
}

static int
read_data_file(
			   FEATUREIO &fio,
			   char *pcFileName
			   )
{
	FILE *infile = fopen( pcFileName, "rb" );
	if( !infile )
	{
		return 0;
	}

	int iDataSizeFloat = fio.x*fio.y*fio.z*fio.t*fio.iFeaturesPerVector;
	assert( iDataSizeFloat > 0 );

	int iDataRead =
		fread( fio.pfVectors, sizeof(float), iDataSizeFloat, infile );

	if( iDataRead != iDataSizeFloat )
	{
		fclose( infile );
		return 0;
	}

	fclose( infile );

	return 1;
}

static int
write_data_file(
			   FEATUREIO &fio,
			   char *pcFileName
			   )
{
	FILE *outfile = fopen( pcFileName, "wb" );
	if( !outfile )
	{
		return 0;
	}

	int iDataSizeFloat = fio.x*fio.y*fio.z*fio.t*fio.iFeaturesPerVector;

	int iFloatsWritten = fwrite( fio.pfVectors, sizeof(float), iDataSizeFloat, outfile );
	float *array_h=static_cast<float *>(fio.pfVectors);
  cudaMemcpy(fio.d_pfVectors, array_h, iDataSizeFloat, cudaMemcpyHostToDevice);
	if( iFloatsWritten != iDataSizeFloat )
	{
		fclose( outfile );
		return 0;
	}

	fclose( outfile );
	return 1;
}

int
fioAllocate(
		FEATUREIO &fio
		)
{
	if( fio.x <= 0 || fio.y <= 0 || fio.z <= 0 || fio.t <= 0 || fio.iFeaturesPerVector <= 0 )
	{
		return 0;
	}

	fio.pfMeans = 0;
	fio.pfVarrs = 0;
	fio.pfVectors = 0;
	fio.d_pfVectors = 0;

	unsigned int iDataSizeFloat = fio.x*fio.y*fio.z*fio.t*fio.iFeaturesPerVector;
	assert( iDataSizeFloat > 0 );

	fio.pfVectors = new float[iDataSizeFloat];
	cudaMalloc((void**)&fio.d_pfVectors, iDataSizeFloat*sizeof(float));

	if( !fio.pfVectors )
	{
		return 0;
	}

	if( !fio.d_pfVectors )
	{
		return 0;
	}

	fio.pfMeans = new float[fio.iFeaturesPerVector];
	if( !fio.pfMeans )
	{
		delete [] fio.pfVectors;
		cudaFree(fio.d_pfVectors);
		return 0;
	}

	fio.pfVarrs = new float[fio.iFeaturesPerVector];
	if( !fio.pfVarrs )
	{
		delete [] fio.pfMeans;
		delete [] fio.pfVectors;
		cudaFree(fio.d_pfVectors);
		return 0;
	}

	return 1;
}


int
fioAllocateExample(
		FEATUREIO &fio,
		const FEATUREIO &fioExample
		)
{
	fio = fioExample;
	fio.pfMeans   = 0;
	fio.pfVarrs   = 0;
	fio.pfVectors = 0;

	return fioAllocate( fio );
}

int
fioRead(
		FEATUREIO &fio,
		char *pcName
		)
{
	char pcFileName[400];
	char pcDataFileName[400];
	pcDataFileName[0] = 0;

	int iReturn = read_info_file( fio, pcName, pcDataFileName );
	if( iReturn == 0 )
	{
		// Attempt to read old *.txt/*.bin format
		sprintf( pcFileName, "%s.txt", pcName );
		sprintf( pcDataFileName, "%s.bin", pcName );

		int iReturn2 = read_info_file( fio, pcFileName, pcDataFileName );
		if( iReturn2 == 0 )
		{
			// Return the original error code from read_info_file()
			return 0;
		}
	}

	// Check if we need to add relative path to file name found header/info
	char *pch = 0;
	if( (pch = strrchr( pcName, '\\' )) && !strrchr( pcDataFileName, '\\' ) )
	{
		strncpy( pcFileName, pcDataFileName, sizeof(pcFileName) );
		sprintf( pcDataFileName, "%s", pcName );
		pch = strrchr( pcDataFileName, '\\' );
		assert( pch );
		pch++;
		sprintf( pch, "%s", pcFileName );
	}
	else if( (pch = strrchr( pcName, '/' )) && !strrchr( pcDataFileName, '/' ) )
	{
		strncpy( pcFileName, pcDataFileName, sizeof(pcFileName) );
		sprintf( pcDataFileName, "%s", pcName );
		pch = strrchr( pcDataFileName, '/' );
		assert( pch );
		pch++;
		sprintf( pch, "%s", pcFileName );
	}

	if( !fioAllocate( fio ) )
	{
		return 0;
	}

	//sprintf( pcFileName, "%s.txt", pcName );
	//if( !read_info_file_data( fio, pcFileName ) )
	//{
	//	return 0;
	//}

	if( !read_data_file( fio, pcDataFileName ) )
	{
		char *pch = strrchr( pcDataFileName, '.' );
		if( !pch )
			return 0;

		if( pch != strstr( pch, ".bin" ) )
			return 0;

		pch[1] = 'i'; pch[2] = 'm'; pch[3] = 'g';
		if( !read_data_file( fio, pcDataFileName ) )
		{
			return 0;
		}
	}

	return 1;
}

int
fioDelete(
		FEATUREIO &fio
		)
{
	if( fio.pfVectors )
	{
		delete [] fio.pfVarrs;
		delete [] fio.pfMeans;
		delete [] fio.pfVectors;
		cudaFree(fio.d_pfVectors);
		fio.pfVarrs=0;
		fio.pfMeans=0;
		fio.pfVectors=0;
		fio.d_pfVectors=0;
	}
	return 1;
}

int
fioTraverseFIO(
			   FEATUREIO &fio,
			   POINT_FUNC func,
			   void *pData
			   )
{
	assert( fio.t == 1 );
	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex = z*fio.y*fio.x*fio.iFeaturesPerVector;
		for( int y = 0; y < fio.y; y++ )
		{
			int iYIndex = iZIndex + y*fio.x*fio.iFeaturesPerVector;
			for( int x = 0; x < fio.x; x++ )
			{
				int iXIndex = iYIndex + x*fio.iFeaturesPerVector;

				int iReturn = func( 1, x, y, z, pData );

				if( iReturn != 0 )
				{
					return iReturn;
				}
			}
		}
	}
	return 0;
}

int
fioModify(
		FEATUREIO &fio,
		int *piRemove,
		int bDiff
		)
{
	float *pfOriginal = fio.pfVectors;
	float *pfModified = fio.pfVectors;

	assert( fio.t == 0 );

	int r = 0;
	int iRemoveCount = 0;

	// Count how many non-zero mappings
	while( piRemove[iRemoveCount] > 0 && iRemoveCount < fio.iFeaturesPerVector )
	{
		iRemoveCount++;
	}

	int iModifIncrement = ( bDiff ? iRemoveCount - 1 : iRemoveCount );

	assert( iModifIncrement >= 0 );

	for( int i = 0; i < fio.z*fio.y*fio.x; i++ )
	{
		// Remove
		for( r = 0; r < iRemoveCount; r++ )
		{
			pfModified[r] = pfOriginal[ piRemove[r] ];
		}

		// pfModified now points to a vector of length iGoodCount

		// Difference
		if( bDiff )
		{
			for( r = 0; r < iModifIncrement; r++ )
			{
				pfModified[r] = pfModified[r] - pfModified[r+1];
			}
		}

		pfOriginal += fio.iFeaturesPerVector;
		pfModified += iModifIncrement;
	}

	fio.iFeaturesPerVector = iModifIncrement;

	return 1;
}




int
fioEstimateVarriance(
					 FEATUREIO &fio
					 )
{
	if( fio.x*fio.y*fio.z <= 0 )
	{
		return 0;
	}

	for( int k = 0; k < fio.iFeaturesPerVector; k++ )
	{
		fio.pfMeans[k] = 0.0f;
		fio.pfVarrs[k] = 0.0f;
	}

	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex = z*fio.y*fio.x*fio.iFeaturesPerVector;
		for( int y = 0; y < fio.y; y++ )
		{
			int iYIndex = iZIndex + y*fio.x*fio.iFeaturesPerVector;
			for( int x = 0; x < fio.x; x++ )
			{
				int iXIndex = iYIndex + x*fio.iFeaturesPerVector;
				for( int k = 0; k < fio.iFeaturesPerVector; k++ )
				{
					float fValue = fio.pfVectors[iXIndex + k];
					fio.pfMeans[k] += fValue;
					fio.pfVarrs[k] += fValue*fValue;
				}
			}
		}
	}

	for( int k = 0; k < fio.iFeaturesPerVector; k++ )
	{
		fio.pfMeans[k] = fio.pfMeans[k] / (float)(fio.x*fio.y);
		fio.pfVarrs[k] = (fio.pfVarrs[k] - fio.pfMeans[k]*fio.pfMeans[k]*(float)(fio.x*fio.y)) / (float)(fio.x*fio.y - 1);
	}
	return 1;
}

int
fioSet(
		FEATUREIO &fio,
		float fValue
		)
{
	if( fio.x*fio.y*fio.z*fio.iFeaturesPerVector <= 0 )
	{
		return 0;
	}

	for( int i = 0; i < fio.x*fio.y*fio.z*fio.iFeaturesPerVector; i++ )
	{
		fio.pfVectors[i] = fValue;
	}

	return 0;
}

int
fioFadeGaussianWindow(
		FEATUREIO &fio,
		float *pfXYZ,
		float fSigma
		)
{
	if( fSigma <= 0 )
	{
		return -1;
	}

	float fMin, fMax;
	int iDataSizeFloat = fio.x*fio.y*fio.z*fio.t*fio.iFeaturesPerVector;

	find_min_max( fio.pfVectors, iDataSizeFloat, fMin, fMax );

	float fVarDiv = 1.0/(2.0*fSigma*fSigma);
	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex = z*fio.y*fio.x*fio.iFeaturesPerVector;
		for( int y = 0; y < fio.y; y++ )
		{
			int iYIndex = iZIndex + y*fio.x*fio.iFeaturesPerVector;
			for( int x = 0; x < fio.x; x++ )
			{
				int iXIndex = iYIndex + x*fio.iFeaturesPerVector;

				float dx = x-pfXYZ[0];
				float dy = y-pfXYZ[1];
				float dz = z-pfXYZ[2];
				float distSqr = dx*dx + dy*dy + dz*dz;

				float fGauss = exp( -distSqr * fVarDiv );

				for( int k = 0; k < fio.iFeaturesPerVector; k++ )
				{
					float fValue = fio.pfVectors[iXIndex + k];
					fio.pfVectors[iXIndex + k] = fGauss*fValue + (1.0f-fGauss)*fMin;
				}
			}
		}
	}
	return 0;
}

float
fioGetPixel(
		const FEATUREIO &fio,
		int x, int y, int z, int t
			)
{
	return fio.pfVectors[ ((t*fio.z + z)*fio.y + y)*fio.x + x ];
}

float *
fioGetVector(
		const FEATUREIO &fio,
		int x, int y, int z, int t
			)
{
	return fio.pfVectors + (((t*fio.z + z)*fio.y + y)*fio.x + x)*fio.iFeaturesPerVector;
}

//
// _fioDetermineInterpCoord()
//
// Determines the coordinate & weight for linear interpolation.
// *Note: the center of a pixel is 0.5. Interpolation of 3.5 will put 100% of weight on pixel 3, 0% on pixel 4.
//
void
_fioDetermineInterpCoord(
				   float fX,
				   float fMinX,
				   float fMaxX,
				   int &iXCoord,
				   float &fXWeight
				   )
{
	if( fX < fMinX + 0.5f )
	{
		iXCoord = fMinX;
		fXWeight = 1.0f;
	}
	else if( fX >= fMaxX - 0.5f )
	{
		iXCoord = fMaxX - 2;
		fXWeight = 0.0f;
	}
	else
	{
		float fMinusHalf = fX - 0.5f;
		iXCoord = (int)floor( fMinusHalf );
		fXWeight = 1.0f - (fMinusHalf - ((float)iXCoord));
	}
}

float
fioGetPixelBilinearInterp(
						   const FEATUREIO &fio,
						   float x, float y
						   )
{
	float fXCont;
	float fYCont;

	int iX;
	int iY;

	_fioDetermineInterpCoord( x, 0, fio.x, iX, fXCont );
	_fioDetermineInterpCoord( y, 0, fio.y, iY, fYCont );

	float f000 = fioGetPixel( fio, iX+0, iY+0, 0 );
	float f100 = fioGetPixel( fio, iX+1, iY+0, 0 );
	float f010 = fioGetPixel( fio, iX+0, iY+1, 0 );
	float f110 = fioGetPixel( fio, iX+1, iY+1, 0 );

	float fn00 = fXCont*f000 + (1.0f - fXCont)*f100;
	float fn10 = fXCont*f010 + (1.0f - fXCont)*f110;

	float fnn0 = fYCont*fn00 + (1.0f - fYCont)*fn10;

	return fnn0;
}

float
fioGetPixelTrilinearInterp(
						   const FEATUREIO &fio,
						   float x, float y, float z
						   )
{
	float fXCont;
	float fYCont;
	float fZCont;

	int iX;
	int iY;
	int iZ;

	_fioDetermineInterpCoord( x, 0, fio.x, iX, fXCont );
	_fioDetermineInterpCoord( y, 0, fio.y, iY, fYCont );
	_fioDetermineInterpCoord( z, 0, fio.z, iZ, fZCont );

	float f000 = fioGetPixel( fio, iX+0, iY+0, iZ+0 );
	float f100 = fioGetPixel( fio, iX+1, iY+0, iZ+0 );
	float f010 = fioGetPixel( fio, iX+0, iY+1, iZ+0 );
	float f110 = fioGetPixel( fio, iX+1, iY+1, iZ+0 );
	float f001 = fioGetPixel( fio, iX+0, iY+0, iZ+1 );
	float f101 = fioGetPixel( fio, iX+1, iY+0, iZ+1 );
	float f011 = fioGetPixel( fio, iX+0, iY+1, iZ+1 );
	float f111 = fioGetPixel( fio, iX+1, iY+1, iZ+1 );

	float fn00 = fXCont*f000 + (1.0f - fXCont)*f100;
	float fn01 = fXCont*f001 + (1.0f - fXCont)*f101;
	float fn10 = fXCont*f010 + (1.0f - fXCont)*f110;
	float fn11 = fXCont*f011 + (1.0f - fXCont)*f111;

	float fnn0 = fYCont*fn00 + (1.0f - fYCont)*fn10;
	float fnn1 = fYCont*fn01 + (1.0f - fYCont)*fn11;

	float fnnn = fZCont*fnn0 + (1.0f - fZCont)*fnn1;

	return fnnn;
}


void
fioIncPixelTrilinearInterp(
						   const FEATUREIO &fio,
						   float x, float y, float z,
						   int iFeature, float fValue
						   )
{
	float fXCont;
	float fYCont;
	float fZCont;

	int iX;
	int iY;
	int iZ;

	_fioDetermineInterpCoord( x, 0, fio.x, iX, fXCont );
	_fioDetermineInterpCoord( y, 0, fio.y, iY, fYCont );
	_fioDetermineInterpCoord( z, 0, fio.z, iZ, fZCont );

	float *pf000 = fioGetVector( fio, iX+0, iY+0, iZ+0 );
	float *pf100 = fioGetVector( fio, iX+1, iY+0, iZ+0 );
	float *pf010 = fioGetVector( fio, iX+0, iY+1, iZ+0 );
	float *pf110 = fioGetVector( fio, iX+1, iY+1, iZ+0 );
	float *pf001 = fioGetVector( fio, iX+0, iY+0, iZ+1 );
	float *pf101 = fioGetVector( fio, iX+1, iY+0, iZ+1 );
	float *pf011 = fioGetVector( fio, iX+0, iY+1, iZ+1 );
	float *pf111 = fioGetVector( fio, iX+1, iY+1, iZ+1 );

	pf000[iFeature] += fValue*fXCont		*fYCont			*fZCont;
	pf100[iFeature] += fValue*(1.0f-fXCont)	*fYCont			*fZCont;
	pf010[iFeature] += fValue*fXCont		*(1.0f-fYCont)	*fZCont;
	pf110[iFeature] += fValue*(1.0f-fXCont)	*(1.0f-fYCont)	*fZCont;
	pf001[iFeature] += fValue*fXCont		*fYCont			*(1.0f-fZCont);
	pf101[iFeature] += fValue*(1.0f-fXCont)	*fYCont			*(1.0f-fZCont);
	pf011[iFeature] += fValue*fXCont		*(1.0f-fYCont)	*(1.0f-fZCont);
	pf111[iFeature] += fValue*(1.0f-fXCont)	*(1.0f-fYCont)	*(1.0f-fZCont);
}

float
fioGetPixelTrilinearInterp(
						   const FEATUREIO &fio,
						   float x, float y, float z, int iFeature
						   )
{
	float fXCont;
	float fYCont;
	float fZCont;

	int iX;
	int iY;
	int iZ;

	_fioDetermineInterpCoord( x, 0, fio.x, iX, fXCont );
	_fioDetermineInterpCoord( y, 0, fio.y, iY, fYCont );
	_fioDetermineInterpCoord( z, 0, fio.z, iZ, fZCont );

	float *pf000 = fioGetVector( fio, iX+0, iY+0, iZ+0 );
	float *pf100 = fioGetVector( fio, iX+1, iY+0, iZ+0 );
	float *pf010 = fioGetVector( fio, iX+0, iY+1, iZ+0 );
	float *pf110 = fioGetVector( fio, iX+1, iY+1, iZ+0 );
	float *pf001 = fioGetVector( fio, iX+0, iY+0, iZ+1 );
	float *pf101 = fioGetVector( fio, iX+1, iY+0, iZ+1 );
	float *pf011 = fioGetVector( fio, iX+0, iY+1, iZ+1 );
	float *pf111 = fioGetVector( fio, iX+1, iY+1, iZ+1 );

	float fn00 = fXCont*pf000[iFeature] + (1.0f - fXCont)*pf100[iFeature];
	float fn01 = fXCont*pf001[iFeature] + (1.0f - fXCont)*pf101[iFeature];
	float fn10 = fXCont*pf010[iFeature] + (1.0f - fXCont)*pf110[iFeature];
	float fn11 = fXCont*pf011[iFeature] + (1.0f - fXCont)*pf111[iFeature];

	float fnn0 = fYCont*fn00 + (1.0f - fYCont)*fn10;
	float fnn1 = fYCont*fn01 + (1.0f - fYCont)*fn11;

	float fnnn = fZCont*fnn0 + (1.0f - fZCont)*fnn1;

	return fnnn;
}

void
fioSetPixelTrilinearInterp(
						   const FEATUREIO &fio,
						   float x, float y, float z,
						   int iFeature, float fValue
						   )
{
	float fXCont;
	float fYCont;
	float fZCont;

	int iX;
	int iY;
	int iZ;

	_fioDetermineInterpCoord( x, 0, fio.x, iX, fXCont );
	_fioDetermineInterpCoord( y, 0, fio.y, iY, fYCont );
	_fioDetermineInterpCoord( z, 0, fio.z, iZ, fZCont );

	float *pf000 = fioGetVector( fio, iX+0, iY+0, iZ+0 );
	float *pf100 = fioGetVector( fio, iX+1, iY+0, iZ+0 );
	float *pf010 = fioGetVector( fio, iX+0, iY+1, iZ+0 );
	float *pf110 = fioGetVector( fio, iX+1, iY+1, iZ+0 );
	float *pf001 = fioGetVector( fio, iX+0, iY+0, iZ+1 );
	float *pf101 = fioGetVector( fio, iX+1, iY+0, iZ+1 );
	float *pf011 = fioGetVector( fio, iX+0, iY+1, iZ+1 );
	float *pf111 = fioGetVector( fio, iX+1, iY+1, iZ+1 );

	pf000[iFeature] = fValue*fXCont		*fYCont			*fZCont;
	pf100[iFeature] = fValue*(1.0f-fXCont)	*fYCont			*fZCont;
	pf010[iFeature] = fValue*fXCont		*(1.0f-fYCont)	*fZCont;
	pf110[iFeature] = fValue*(1.0f-fXCont)	*(1.0f-fYCont)	*fZCont;
	pf001[iFeature] = fValue*fXCont		*fYCont			*(1.0f-fZCont);
	pf101[iFeature] = fValue*(1.0f-fXCont)	*fYCont			*(1.0f-fZCont);
	pf011[iFeature] = fValue*fXCont		*(1.0f-fYCont)	*(1.0f-fZCont);
	pf111[iFeature] = fValue*(1.0f-fXCont)	*(1.0f-fYCont)	*(1.0f-fZCont);
}

int
fioWrite(
		FEATUREIO &fio,
		char *pcName
		)
{
	char pcFileName[200];

	sprintf( pcFileName, "%s.txt", pcName );
	if( !write_info_file( fio, pcFileName ) )
	{
		return 0;
	}

	sprintf( pcFileName, "%s.img", pcName );
	if( !write_data_file( fio, pcFileName ) )
	{
		return 0;
	}

	return 1;
}

int
fioFeatureSliceXY(
		FEATUREIO &fio,
		int iValue,			// Value in Z dimension
		int iFeature,		// Index of feature to output
		float *pfSlice		// Memory for slice
		)
{
	int iZIndex = iValue*fio.x*fio.y;
	if( fio.z <= 1 )
	{
		iZIndex = 0;
	}

	for( int y = 0; y < fio.y; y++ )
	{
		int iYIndex = iZIndex + y*fio.x;

		for( int x = 0; x < fio.x; x++ )
		{
			int iIndex = iYIndex + x;

			pfSlice[ y*fio.x + x ] =
				fio.pfVectors[ iIndex*fio.iFeaturesPerVector + iFeature ];
		}
	}

	//ppImgOut.InitializeSubImage( fio.x, fio.y, fio.iFeaturesPerVector*fio.y*sizeof(float), fio.iFeaturesPerVector*sizeof(float)*8, (unsigned char*)(fio.pfVectors + iValue*fio.x*fio.y*fio.iFeaturesPerVector) );

	return 1;
}

int
fioFeatureSliceZX(
		FEATUREIO &fio,
		int iValue,			// Value in Y dimension
		int iFeature,		// Index of feature to output
		float *pfSlice		// Memory for slice
		)
{
	int iYIndex = iValue*fio.x;
	if( fio.z <= 1 )
	{
		return -1;
	}

	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex = iYIndex + z*fio.x*fio.y;
		for( int x = 0; x < fio.x; x++ )
		{
			int iIndex = iZIndex + x;

			pfSlice[ z*fio.x + x ] =
				fio.pfVectors[ iIndex*fio.iFeaturesPerVector + iFeature ];
		}
	}

	return 0;
}

int
fioFeatureSliceZY(
		FEATUREIO &fio,
		int iValue,			// Value in X dimension
		int iFeature,		// Index of feature to output
		float *pfSlice		// Memory for slice
		)
{
	int iXIndex = iValue;
	if( fio.z <= 1 )
	{
		return -1;
	}

	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex = iXIndex + z*fio.x*fio.y;
		for( int y = 0; y < fio.y; y++ )
		{
			int iIndex = iZIndex + y*fio.x;

			pfSlice[ z*fio.y + y ] =
				fio.pfVectors[ iIndex*fio.iFeaturesPerVector + iFeature ];
		}
	}

	return 0;
}

int
fioExtractSingleFeature(
		FEATUREIO &fioFeatureSingle,
		FEATUREIO &fioFeatureMulti,
		int iFeature
		)
{
	assert( fioFeatureSingle.iFeaturesPerVector == 1 );
	assert( fioFeatureSingle.x == fioFeatureMulti.x );
	assert( fioFeatureSingle.y == fioFeatureMulti.y );
	assert( fioFeatureSingle.z == fioFeatureMulti.z );
	assert( iFeature >= 0 );
	assert( iFeature < fioFeatureMulti.iFeaturesPerVector );

	for( int z = 0; z < fioFeatureSingle.z; z++ )
	{
		int iZIndex = z*fioFeatureSingle.y*fioFeatureSingle.x;

		for( int y = 0; y < fioFeatureSingle.y; y++ )
		{
			int iYIndex = iZIndex + y*fioFeatureSingle.x;

			for( int x = 0; x < fioFeatureSingle.x; x++ )
			{
				int iXIndex = iYIndex + x;

				fioFeatureSingle.pfVectors[iXIndex] =
					fioFeatureMulti.pfVectors[iXIndex*fioFeatureMulti.iFeaturesPerVector + iFeature];
			}
		}
	}
	return 1;
}

int
fioCrop(
		FEATUREIO &fioIn,
		FEATUREIO &fioOut,
		int xin,
		int yin,
		int zin
		)
{
	assert( fioIn.iFeaturesPerVector == fioOut.iFeaturesPerVector );

	if( fioIn.iFeaturesPerVector != fioOut.iFeaturesPerVector )
	{
		return 0;
	}

	for( int z = 0; z < fioOut.z; z++ )
	{
		int iZInIndex  =  (z+zin)*fioIn.x*fioIn.y;
		int iZOutIndex =      z*fioOut.x*fioOut.y;

		for( int y = 0; y < fioOut.y; y++ )
		{
			int iYInIndex  = iZInIndex + (y+yin)*fioIn.x;
			int iYOutIndex = iZOutIndex + y*fioOut.x;

			for( int x = 0; x < fioOut.x; x++ )
			{
				int iInIndex = (iYInIndex + x+xin)*fioIn.iFeaturesPerVector;
				int iOutIndex = (iYOutIndex + x)*fioIn.iFeaturesPerVector;

				memcpy( fioOut.pfVectors + iOutIndex,
					fioIn.pfVectors + iInIndex,
					fioOut.iFeaturesPerVector*sizeof(float)
					);
			}
		}
	}

	return 1;
}


int
fioFindMin(
		   FEATUREIO &fio,
		   int &ix,
		   int &iy,
		   int &iz,
		   float &fMin
		   )
{
	ix = 0; iy = 0; iz = 0;
	fMin = fio.pfVectors[0];
	for( int z = 0; z < fio.z; z++ )
	{
		for( int y = 0; y < fio.y; y++ )
		{
			for( int x = 0; x < fio.x; x++ )
			{
				float fPix = fioGetPixel( fio, x, y, z );
				if( fPix < fMin )
				{
					fMin = fPix;
					ix = x; iy = y; iz = z;
				}
			}
		}
	}
	return 0;
}

int
fioFindMax(
		   FEATUREIO &fio,
		   int &ix,
		   int &iy,
		   int &iz,
		   float &fMax
		   )
{
	ix = 0; iy = 0; iz = 0;
	fMax = fio.pfVectors[0];
	for( int z = 0; z < fio.z; z++ )
	{
		for( int y = 0; y < fio.y; y++ )
		{
			for( int x = 0; x < fio.x; x++ )
			{
				float fPix = fioGetPixel( fio, x, y, z );
				if( fPix > fMax )
				{
					fMax = fPix;
					ix = x; iy = y; iz = z;
				}
			}
		}
	}
	return 0;
}

int
fioFindMin(
		   FEATUREIO &fio,
		   int &ix,
		   int &iy,
		   int &iz
		   )
{
	float fValue;
	return fioFindMin( fio, ix, iy, iz, fValue );
}


int
fioFindMax(
		   FEATUREIO &fio,
		   int &ix,
		   int &iy,
		   int &iz
		   )
{
	float fValue;
	return fioFindMax( fio, ix, iy, iz, fValue );
}

int
fioFindMin(
		   FEATUREIO &fio,
		   float &fValue
		   )
{
   int ix;
   int iy;
   int iz;
   return fioFindMin( fio, ix, iy, iz, fValue );
}


int
fioFindMax(
		   FEATUREIO &fio,
		   float &fValue
		   )
{
   int ix;
   int iy;
   int iz;
   return fioFindMax( fio, ix, iy, iz, fValue );
}

int
fioZero(
		FEATUREIO &fio
		)
{
	fio.iFeaturesPerVector = 0;
	fio.pfMeans = 0;
	fio.pfVarrs = 0;
	fio.pfVectors = 0;
	fio.t = 0;
	fio.x = 0;
	fio.y = 0;
	fio.z = 0;
	return 0;
}

int
fioFindMinMax(
		FEATUREIO &fio,
			 float &fMin,			// Min value
			 float &fMax				// Max value
			 )
{
	int iFeats = fio.x*fio.y*fio.z*fio.t*fio.iFeaturesPerVector;
	return find_min_max( fio.pfVectors, iFeats, fMin, fMax );
}

int
fioNormalizeSquaredLength(
						  FEATUREIO &fio
						  )
{
	float fSumSqrs = 0;

	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex = z*fio.y*fio.x*fio.iFeaturesPerVector;
		for( int y = 0; y < fio.y; y++ )
		{
			int iYIndex = iZIndex + y*fio.x*fio.iFeaturesPerVector;
			for( int x = 0; x < fio.x; x++ )
			{
				int iXIndex = iYIndex + x*fio.iFeaturesPerVector;
				for( int k = 0; k < fio.iFeaturesPerVector; k++ )
				{
					float fValue = fio.pfVectors[iXIndex + k];
					fSumSqrs += fValue*fValue;
				}
			}
		}
	}

	if( fSumSqrs <=  0 )
	{
		return -1;
	}

	float fDiv = 1.0f / sqrt( fSumSqrs );
	fSumSqrs = 0;

	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex = z*fio.y*fio.x*fio.iFeaturesPerVector;
		for( int y = 0; y < fio.y; y++ )
		{
			int iYIndex = iZIndex + y*fio.x*fio.iFeaturesPerVector;
			for( int x = 0; x < fio.x; x++ )
			{
				int iXIndex = iYIndex + x*fio.iFeaturesPerVector;
				for( int k = 0; k < fio.iFeaturesPerVector; k++ )
				{
					float fValue = fio.pfVectors[iXIndex + k];
					fValue *= fDiv;
					fio.pfVectors[iXIndex + k] = fValue;
					fSumSqrs += fValue*fValue;
				}
			}
		}
	}

	return 1;
}

int
fioNormalizeSquaredLengthZeroMean(
						  FEATUREIO &fio
						  )
{
	float fSum = 0;
	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex = z*fio.y*fio.x*fio.iFeaturesPerVector;
		for( int y = 0; y < fio.y; y++ )
		{
			int iYIndex = iZIndex + y*fio.x*fio.iFeaturesPerVector;
			for( int x = 0; x < fio.x; x++ )
			{
				int iXIndex = iYIndex + x*fio.iFeaturesPerVector;
				for( int k = 0; k < fio.iFeaturesPerVector; k++ )
				{
					float fValue = fio.pfVectors[iXIndex + k];
					fSum += fValue;
				}
			}
		}
	}

	fSum /= fio.z*fio.y*fio.x*fio.iFeaturesPerVector;
	float fSumSqrs = 0;

	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex = z*fio.y*fio.x*fio.iFeaturesPerVector;
		for( int y = 0; y < fio.y; y++ )
		{
			int iYIndex = iZIndex + y*fio.x*fio.iFeaturesPerVector;
			for( int x = 0; x < fio.x; x++ )
			{
				int iXIndex = iYIndex + x*fio.iFeaturesPerVector;
				for( int k = 0; k < fio.iFeaturesPerVector; k++ )
				{
					float fValue = fio.pfVectors[iXIndex + k];
					fValue -= fSum;
					fSumSqrs += fValue*fValue;
				}
			}
		}
	}

	if( fSumSqrs <=  0 )
	{
		return -1;
	}

	float fDiv = 1.0f / sqrt( fSumSqrs );
	fSumSqrs = 0;

	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex = z*fio.y*fio.x*fio.iFeaturesPerVector;
		for( int y = 0; y < fio.y; y++ )
		{
			int iYIndex = iZIndex + y*fio.x*fio.iFeaturesPerVector;
			for( int x = 0; x < fio.x; x++ )
			{
				int iXIndex = iYIndex + x*fio.iFeaturesPerVector;
				for( int k = 0; k < fio.iFeaturesPerVector; k++ )
				{
					float fValue = fio.pfVectors[iXIndex + k];
					fValue -= fSum;
					fValue *= fDiv;
					fio.pfVectors[iXIndex + k] = fValue;
					fSumSqrs += fValue*fValue;
				}
			}
		}
	}

	return 1;
}


int
fioNormalize(
		FEATUREIO &fio,
		const float &fStretch
		)
{
	int i;
	int iFeatures = fio.x*fio.y*fio.z;

	assert( fio.iFeaturesPerVector == 1 );
	if( fio.iFeaturesPerVector != 1 )
	{
		return 0;
	}

	float fMin;
	float fMax;

	find_min_max( fio.pfVectors, iFeatures, fMin, fMax );

	if( fMax == fMin )
	{
		// All black image
		for( int i = 0; i < iFeatures; i++ )
		{
			fio.pfVectors[i] = 0;
		}
	}
	else
	{
		assert( fMax > fMin);
		float fRange = fStretch / (fMax - fMin);

		for( int i = 0; i < iFeatures; i++ )
		{
			float fValue = fio.pfVectors[i];
			fValue -= fMin;
			fValue *= fRange;
			fio.pfVectors[i] = fValue;
		}
	}

	return 1;
}

int
fioSubSampleInterpolate(
		FEATUREIO &fioIn,
		FEATUREIO &fioOut
		)
{
	assert( fioIn.iFeaturesPerVector == fioOut.iFeaturesPerVector );

	if( fioIn.iFeaturesPerVector != fioOut.iFeaturesPerVector )
	{
		return 0;
	}

	if( fioOut.z > 1 )
	{
		assert( fioIn.x/2 == fioOut.x &&
			fioIn.y/2 == fioOut.y &&
			fioIn.z/2 == fioOut.z );
	}
	else
	{
		assert( fioIn.x/2 == fioOut.x &&
			fioIn.y/2 == fioOut.y );
	}

	for( int z = 0; z < fioOut.z; z++ )
	{
		for( int y = 0; y < fioOut.y; y++ )
		{
			for( int x = 0; x < fioOut.x; x++ )
			{
				float *pfVectorOut = fioGetVector( fioOut, x, y, z );

				float *pf000;
				float *pf100;
				float *pf010;
				float *pf110;

				float *pf001;
				float *pf101;
				float *pf011;
				float *pf111;

				pf000 = fioGetVector( fioIn, 2*x+0, 2*y+0, 2*z+0 );
				pf100 = fioGetVector( fioIn, 2*x+1, 2*y+0, 2*z+0 );
				pf010 = fioGetVector( fioIn, 2*x+0, 2*y+1, 2*z+0 );
				pf110 = fioGetVector( fioIn, 2*x+1, 2*y+1, 2*z+0 );

				if( 2*z+1 < fioIn.z )
				{
					pf001 = fioGetVector( fioIn, 2*x+0, 2*y+0, 2*z+1 );
					pf101 = fioGetVector( fioIn, 2*x+1, 2*y+0, 2*z+1 );
					pf011 = fioGetVector( fioIn, 2*x+0, 2*y+1, 2*z+1 );
					pf111 = fioGetVector( fioIn, 2*x+1, 2*y+1, 2*z+1 );
				}

				for( int k = 0; k < fioIn.iFeaturesPerVector; k++ )
				{
					float fSum = 0;
					fSum += pf000[k] + pf010[k] + pf100[k] + pf110[k];
					if( 2*z+1 < fioIn.z )
					{
						fSum += pf001[k] + pf011[k] + pf101[k] + pf111[k];
						fSum *= 0.125;
					}
					else
					{
						fSum *= 0.25;
					}

					pfVectorOut[k] = fSum;
				}
			}
		}
	}
	int iDataSizeFloat=fioOut.x*fioOut.y*fioOut.z*fioOut.t*fioOut.iFeaturesPerVector*sizeof(float);
	float *array_h=static_cast<float *>(fioOut.pfVectors);
	cudaMemcpy(fioOut.d_pfVectors, array_h, iDataSizeFloat, cudaMemcpyHostToDevice);
	return 1;
}

int
Subsample_interleave(FEATUREIO &fioG2, FEATUREIO &fioSaveHalf, int best_device_id){
	if (best_device_id) {
		return SubSampleInterpolateCuda(fioG2, fioSaveHalf, best_device_id);
	}
	else{
		return fioSubSampleInterpolate(fioG2, fioSaveHalf);
	}
}

int
fioSubSample(
		FEATUREIO &fioIn,
		FEATUREIO &fioOut
		)
{
	assert( fioIn.iFeaturesPerVector == fioOut.iFeaturesPerVector );

	if( fioIn.iFeaturesPerVector != fioOut.iFeaturesPerVector )
	{
		return 0;
	}

	//
	// Instead of asserting limits, we simply copy what is possible in both images
	//
	//if( fioOut.z > 1 )
	//{
	//	assert( fioIn.x/2 == fioOut.x &&
	//		fioIn.y/2 == fioOut.y &&
	//		fioIn.z/2 == fioOut.z );
	//}
	//else
	//{
	//	assert( fioIn.x/2 == fioOut.x &&
	//		fioIn.y/2 == fioOut.y );
	//}

	fioSet( fioOut, 0 );
	for( int z = 0; z < fioOut.z && 2*z < fioIn.z; z++ )
	{
		int iZInIndex  =  2*(z*fioIn.x*fioIn.y);
		int iZOutIndex =  z*fioOut.x*fioOut.y;

		for( int y = 0; y < fioOut.y && 2*y < fioIn.y; y++ )
		{
			int iYInIndex  = iZInIndex + 2*(y*fioIn.x);
			int iYOutIndex = iZOutIndex + y*fioOut.x;

			for( int x = 0; x < fioOut.x && 2*x < fioIn.x; x++ )
			{
				int iXInIndex  = iYInIndex + 2*(x);
				int iXOutIndex = iYOutIndex + x;

				float *pfInIndex  = fioIn.pfVectors + iXInIndex*fioIn.iFeaturesPerVector;
				float *pfOutIndex = fioOut.pfVectors + iXOutIndex*fioOut.iFeaturesPerVector;

				memcpy( pfOutIndex, pfInIndex,
						fioOut.iFeaturesPerVector*sizeof(float) );
			}
		}
	}
	int iDataSizeFloat = fioOut.x*fioOut.y*fioOut.z*fioOut.t*fioOut.iFeaturesPerVector*sizeof(float);
	cudaMemcpy(fioOut.d_pfVectors, fioOut.pfVectors, iDataSizeFloat, cudaMemcpyHostToDevice);
	return 1;
}

int
fioSubSample2D(
		FEATUREIO &fioIn,
		FEATUREIO &fioOut
		)
{
	assert( fioIn.z == 1 );

	assert( fioIn.iFeaturesPerVector == fioOut.iFeaturesPerVector );

	if( fioIn.iFeaturesPerVector != fioOut.iFeaturesPerVector )
	{
		return 0;
	}

	assert( fioIn.x/2 == fioOut.x &&
			fioIn.y/2 == fioOut.y &&
			fioIn.z == fioOut.z);

	for( int z = 0; z < fioOut.z; z++ )
	{
		int iZInIndex  =  z*fioIn.x*fioIn.y;
		int iZOutIndex =  z*fioOut.x*fioOut.y;

		for( int y = 0; y < fioOut.y; y++ )
		{
			int iYInIndex  = iZInIndex + 2*(y*fioIn.x);
			int iYOutIndex = iZOutIndex + y*fioOut.x;

			for( int x = 0; x < fioOut.x; x++ )
			{
				int iXInIndex  = iYInIndex + 2*(x);
				int iXOutIndex = iYOutIndex + x;

				float *pfInIndex  = fioIn.pfVectors + iXInIndex*fioIn.iFeaturesPerVector;
				float *pfOutIndex = fioOut.pfVectors + iXOutIndex*fioOut.iFeaturesPerVector;

				memcpy( pfOutIndex, pfInIndex,
						fioOut.iFeaturesPerVector*sizeof(float) );
			}
		}
	}

	return 1;
}

int
fioSubSample2DCenterPixel(
		FEATUREIO &fioIn,
		FEATUREIO &fioOut
		)
{
	//assert( fioIn.z == 1 );

	assert( fioOut.iFeaturesPerVector == 1 );
	assert( fioIn.iFeaturesPerVector  == 1 );

	if( fioIn.iFeaturesPerVector != fioOut.iFeaturesPerVector )
	{
		return 0;
	}

	assert( fioIn.x >= 2*fioOut.x &&
			fioIn.y >= 2*fioOut.y &&
			(fioIn.z >= fioOut.z || fioIn.z == 1)
			);

	for( int z = 0; z < fioOut.z; z++ )
	{
		for( int y = 0; y < fioOut.y; y++ )
		{
			for( int x = 0; x < fioOut.x; x++ )
			{
				float *pfGetVector = fioGetVector( fioOut, x, y, z );
				float fValue=0;
				fValue += fioGetPixel( fioIn, 2*x+0, 2*y+0, 2*z+0 );
				fValue += fioGetPixel( fioIn, 2*x+0, 2*y+0, 2*z+1 );
				fValue += fioGetPixel( fioIn, 2*x+0, 2*y+1, 2*z+0 );
				fValue += fioGetPixel( fioIn, 2*x+0, 2*y+1, 2*z+1 );
				fValue += fioGetPixel( fioIn, 2*x+1, 2*y+0, 2*z+0 );
				fValue += fioGetPixel( fioIn, 2*x+1, 2*y+0, 2*z+1 );
				fValue += fioGetPixel( fioIn, 2*x+1, 2*y+1, 2*z+0 );
				fValue += fioGetPixel( fioIn, 2*x+1, 2*y+1, 2*z+1 );
				*pfGetVector = fValue/8.0f;

			}
		}
	}

	return 1;
}

int
fioMultiply(
		FEATUREIO &fioIn1,
		FEATUREIO &fioIn2,
		FEATUREIO &fioOut
		)
{
	if(
		fioIn1.x*fioIn1.y*fioIn1.z*fioIn1.iFeaturesPerVector != fioIn2.x*fioIn2.y*fioIn2.z*fioIn2.iFeaturesPerVector
		||
		fioIn1.x*fioIn1.y*fioIn1.z*fioIn1.iFeaturesPerVector != fioOut.x*fioOut.y*fioOut.z*fioOut.iFeaturesPerVector
		||
		fioIn2.x*fioIn2.y*fioIn2.z*fioIn2.iFeaturesPerVector != fioOut.x*fioOut.y*fioOut.z*fioOut.iFeaturesPerVector
		)
	{
		return 0;
	}
	for( int z = 0; z < fioOut.z; z++ )
	{
		int iZIndex  =  z*fioOut.y*fioOut.x*fioOut.iFeaturesPerVector;

		for( int y = 0; y < fioOut.y; y++ )
		{
			int iYIndex  = iZIndex + y*fioOut.x*fioOut.iFeaturesPerVector;

			for( int x = 0; x < fioOut.x; x++ )
			{
				int iXIndex  = iYIndex + x*fioOut.iFeaturesPerVector;
				for( int f = 0; f < fioOut.iFeaturesPerVector; f++ )
				{
					fioOut.pfVectors[iXIndex + f] = fioIn1.pfVectors[iXIndex + f]*fioIn2.pfVectors[iXIndex + f];
				}
			}
		}
	}
	return 1;
}

int
fioMin(
		FEATUREIO &fioIn1,
		FEATUREIO &fioIn2,
		FEATUREIO &fioOut
		)
{
	if(
		fioIn1.x*fioIn1.y*fioIn1.z*fioIn1.iFeaturesPerVector != fioIn2.x*fioIn2.y*fioIn2.z*fioIn2.iFeaturesPerVector
		||
		fioIn1.x*fioIn1.y*fioIn1.z*fioIn1.iFeaturesPerVector != fioOut.x*fioOut.y*fioOut.z*fioOut.iFeaturesPerVector
		||
		fioIn2.x*fioIn2.y*fioIn2.z*fioIn2.iFeaturesPerVector != fioOut.x*fioOut.y*fioOut.z*fioOut.iFeaturesPerVector
		)
	{
		return 0;
	}
	for( int z = 0; z < fioOut.z; z++ )
	{
		int iZIndex  =  z*fioOut.y*fioOut.x*fioOut.iFeaturesPerVector;

		for( int y = 0; y < fioOut.y; y++ )
		{
			int iYIndex  = iZIndex + y*fioOut.x*fioOut.iFeaturesPerVector;

			for( int x = 0; x < fioOut.x; x++ )
			{
				int iXIndex  = iYIndex + x*fioOut.iFeaturesPerVector;
				for( int f = 0; f < fioOut.iFeaturesPerVector; f++ )
				{
					fioOut.pfVectors[iXIndex + f] =
						fioIn1.pfVectors[iXIndex + f] < fioIn2.pfVectors[iXIndex + f] ?
					fioIn1.pfVectors[iXIndex + f] : fioIn2.pfVectors[iXIndex + f];
				}
			}
		}
	}
	return 1;
}


int
fioCopy(
		FEATUREIO &fioDst,
		FEATUREIO &fioSrc,
		int iDstFeature,
		int iSrcFeature
		)
{
	int bDimensionsEqual =
		(fioDst.x == fioSrc.x
		&&
		fioDst.y == fioSrc.y
		&&
		fioDst.z == fioSrc.z);
	if( !bDimensionsEqual )
	{
		return 0;
	}

	for( int z = 0; z < fioDst.z; z++ )
	{
		int iZIndex   =  z*fioDst.y*fioDst.x*fioDst.iFeaturesPerVector;
		int iZIndex2  =  z*fioSrc.y*fioSrc.x*fioSrc.iFeaturesPerVector;

		for( int y = 0; y < fioDst.y; y++ )
		{
			int iYIndex   = iZIndex + y*fioDst.x*fioDst.iFeaturesPerVector;
			int iYIndex2  = iZIndex2 + y*fioSrc.x*fioSrc.iFeaturesPerVector;

			for( int x = 0; x < fioDst.x; x++ )
			{
				int iXIndex   = iYIndex + x*fioDst.iFeaturesPerVector;
				int iXIndex2  = iYIndex2 + x*fioSrc.iFeaturesPerVector;

				fioDst.pfVectors[iXIndex + iDstFeature] = fioSrc.pfVectors[iXIndex2 + iSrcFeature];
				float *array_h=static_cast<float *>(fioSrc.pfVectors);
				int iDataSizeFloat = fioDst.z*fioDst.y*fioDst.x*fioDst.iFeaturesPerVector*fioDst.t*sizeof(float);
				cudaMemcpy(fioDst.d_pfVectors, array_h, iDataSizeFloat, cudaMemcpyHostToDevice);
			}
		}
	}

	return 1;
}

int
fioCopy(
		FEATUREIO &fioDst,
		FEATUREIO &fioSrc
		)
{
	if( !fioDimensionsEqual( fioDst, fioSrc ) )
	{
		return 0;
	}

	memcpy(
		fioDst.pfVectors,
		fioSrc.pfVectors,
		fioDst.t*fioDst.x*fioDst.y*fioDst.z*fioDst.iFeaturesPerVector*sizeof(float)
		);
		float *array_h=static_cast<float *>(fioSrc.pfVectors);
		int iDataSizeFloat = fioDst.z*fioDst.y*fioDst.x*fioDst.iFeaturesPerVector*fioDst.t*sizeof(float);
		cudaMemcpy(fioDst.d_pfVectors, array_h, iDataSizeFloat, cudaMemcpyHostToDevice);

	return 1;
}

int
fioDimensionsEqual(
		FEATUREIO &fio1,
		FEATUREIO &fio2
		)
{
	return
		fio1.iFeaturesPerVector == fio2.iFeaturesPerVector
		&&
		fio1.x == fio2.x
		&&
		fio1.y == fio2.y
		&&
		fio1.z == fio2.z;
}

float
fioSum(
	   FEATUREIO &fio,
	   int iFeature
	   )
{
	float fSum = 0.0f;
	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex  =  z*fio.y*fio.x*fio.iFeaturesPerVector;

		for( int y = 0; y < fio.y; y++ )
		{
			int iYIndex  = iZIndex + y*fio.x*fio.iFeaturesPerVector;

			for( int x = 0; x < fio.x; x++ )
			{
				int iXIndex  = iYIndex + x*fio.iFeaturesPerVector;
				fSum += fio.pfVectors[iXIndex + iFeature];
			}
		}
	}
	return fSum;
}

int
fioAbs(
	   FEATUREIO &fio
	   )
{
	for( int z = 0; z < fio.z; z++ )
	{
		int iZIndex  =  z*fio.y*fio.x*fio.iFeaturesPerVector;

		for( int y = 0; y < fio.y; y++ )
		{
			int iYIndex  = iZIndex + y*fio.x*fio.iFeaturesPerVector;

			for( int x = 0; x < fio.x; x++ )
			{
				int iXIndex  = iYIndex + x*fio.iFeaturesPerVector;
				for( int k = 0; k < fio.iFeaturesPerVector; k++ )
				{
					fio.pfVectors[iXIndex + k] = (float)fabs( fio.pfVectors[iXIndex + k] );
				}
			}
		}
	}
	return 1;
}

int
fioMultSum_interleave(
		FEATUREIO &fioIn1,
		FEATUREIO &fioIn2,
		FEATUREIO &fioOut,
		const float &fMultIn2,
		int best_device_id
		)
{
	if (best_device_id>0) {
		return fioCudaMultSum(fioIn1,fioIn2,fioOut,fMultIn2);
	}
	else{
		return fioMultSum(fioIn1,fioIn2,fioOut,fMultIn2);
	}
}


int
fioMultSum(
		FEATUREIO &fioIn1,
		FEATUREIO &fioIn2,
		FEATUREIO &fioOut,
		const float &fMultIn2
		)
{
	if(
		fioIn1.x*fioIn1.y*fioIn1.z*fioIn1.iFeaturesPerVector != fioIn2.x*fioIn2.y*fioIn2.z*fioIn2.iFeaturesPerVector
		||
		fioIn1.x*fioIn1.y*fioIn1.z*fioIn1.iFeaturesPerVector != fioOut.x*fioOut.y*fioOut.z*fioOut.iFeaturesPerVector
		||
		fioIn2.x*fioIn2.y*fioIn2.z*fioIn2.iFeaturesPerVector != fioOut.x*fioOut.y*fioOut.z*fioOut.iFeaturesPerVector
		)
	{
		return 0;
	}
	for( int z = 0; z < fioOut.z; z++ )
	{
		int iZIndex  =  z*fioOut.y*fioOut.x*fioOut.iFeaturesPerVector;

		for( int y = 0; y < fioOut.y; y++ )
		{
			int iYIndex  = iZIndex + y*fioOut.x*fioOut.iFeaturesPerVector;

			for( int x = 0; x < fioOut.x; x++ )
			{
				int iXIndex  = iYIndex + x*fioOut.iFeaturesPerVector;
				for( int f = 0; f < fioOut.iFeaturesPerVector; f++ )
				{
					fioOut.pfVectors[iXIndex + f] = fioIn1.pfVectors[iXIndex + f] + fMultIn2*fioIn2.pfVectors[iXIndex + f];
				}
			}
		}
	}
	return 1;
}

int
fioMult(
		FEATUREIO &fioIn1,
		FEATUREIO &fioOut,
		const float &fMultIn
		)
{
	if(
		fioIn1.x*fioIn1.y*fioIn1.z*fioIn1.iFeaturesPerVector != fioOut.x*fioOut.y*fioOut.z*fioOut.iFeaturesPerVector
		)
	{
		return 0;
	}
	for( int z = 0; z < fioOut.z; z++ )
	{
		int iZIndex  =  z*fioOut.y*fioOut.x*fioOut.iFeaturesPerVector;

		for( int y = 0; y < fioOut.y; y++ )
		{
			int iYIndex  = iZIndex + y*fioOut.x*fioOut.iFeaturesPerVector;

			for( int x = 0; x < fioOut.x; x++ )
			{
				int iXIndex  = iYIndex + x*fioOut.iFeaturesPerVector;
				for( int f = 0; f < fioOut.iFeaturesPerVector; f++ )
				{
					fioOut.pfVectors[iXIndex + f] = fMultIn*fioIn1.pfVectors[iXIndex + f];
				}
			}
		}
	}
	return 1;
}

int
fioInitNeighbourOffsets(
			   FEATUREIO &fio,
			   int &iNeighbourCount,
			   int *piNeighbourOffsets,
			   int bDiagonal,
			   int bSymetric
			   )
{
	if( fio.z > 1 )
	{
		// 3D
		if( bDiagonal )
		{
			if( bSymetric )
			{
				iNeighbourCount = 13; //26 / 2;

				piNeighbourOffsets[0] =  1;
				piNeighbourOffsets[1] = fio.x - 1;
				piNeighbourOffsets[2] = fio.x;
				piNeighbourOffsets[3] = fio.x + 1;

				piNeighbourOffsets[4] = fio.x*fio.y - fio.x - 1;
				piNeighbourOffsets[5] = fio.x*fio.y - fio.x;
				piNeighbourOffsets[6] = fio.x*fio.y - fio.x + 1;
				piNeighbourOffsets[7] = fio.x*fio.y - 1;
				piNeighbourOffsets[8] = fio.x*fio.y;
				piNeighbourOffsets[9] = fio.x*fio.y + 1;
				piNeighbourOffsets[10] = fio.x*fio.y + fio.x - 1;
				piNeighbourOffsets[11] = fio.x*fio.y + fio.x;
				piNeighbourOffsets[12] = fio.x*fio.y + fio.x + 1;
			}
			else
			{
				iNeighbourCount = 26;

				piNeighbourOffsets[0]  = -fio.x*fio.y - fio.x - 1;
				piNeighbourOffsets[1]  = -fio.x*fio.y - fio.x;
				piNeighbourOffsets[2]  = -fio.x*fio.y - fio.x + 1;
				piNeighbourOffsets[3]  = -fio.x*fio.y - 1;
				piNeighbourOffsets[4]  = -fio.x*fio.y;
				piNeighbourOffsets[5]  = -fio.x*fio.y + 1;
				piNeighbourOffsets[6]  = -fio.x*fio.y + fio.x - 1;
				piNeighbourOffsets[7]  = -fio.x*fio.y + fio.x;
				piNeighbourOffsets[8]  = -fio.x*fio.y + fio.x + 1;

				piNeighbourOffsets[9]  = -fio.x - 1;
				piNeighbourOffsets[10] = -fio.x;
				piNeighbourOffsets[11] = -fio.x + 1;
				piNeighbourOffsets[12] = -1;
				piNeighbourOffsets[13] =  1;
				piNeighbourOffsets[14] = fio.x - 1;
				piNeighbourOffsets[15] = fio.x;
				piNeighbourOffsets[16] = fio.x + 1;

				piNeighbourOffsets[17] = fio.x*fio.y - fio.x - 1;
				piNeighbourOffsets[18] = fio.x*fio.y - fio.x;
				piNeighbourOffsets[19] = fio.x*fio.y - fio.x + 1;
				piNeighbourOffsets[20] = fio.x*fio.y - 1;
				piNeighbourOffsets[21] = fio.x*fio.y;
				piNeighbourOffsets[22] = fio.x*fio.y + 1;
				piNeighbourOffsets[23] = fio.x*fio.y + fio.x - 1;
				piNeighbourOffsets[24] = fio.x*fio.y + fio.x;
				piNeighbourOffsets[25] = fio.x*fio.y + fio.x + 1;
			}
		}
		else
		{
			if( bSymetric )
			{
				iNeighbourCount = 3; //6 / 2;

				piNeighbourOffsets[0] =  1;            //  x
				piNeighbourOffsets[1] = fio.x;         //  y
				piNeighbourOffsets[2] = fio.x*fio.y;   //  z
			}
			else
			{
				iNeighbourCount = 6;

				piNeighbourOffsets[0]  = -fio.x*fio.y; // -z
				piNeighbourOffsets[1] = -fio.x;		   // -y
				piNeighbourOffsets[2] = -1;            // -x
				piNeighbourOffsets[3] =  1;            //  x
				piNeighbourOffsets[4] = fio.x;         //  y
				piNeighbourOffsets[5] = fio.x*fio.y;   //  z
			}
		}
	}
	else
	{
		// 2D

		if( bDiagonal )
		{
			if( bSymetric )
			{
				iNeighbourCount = 4; //8 / 2;

				piNeighbourOffsets[0] =  1;
				piNeighbourOffsets[1] = fio.x-1;
				piNeighbourOffsets[2] = fio.x;
				piNeighbourOffsets[3] = fio.x+1;
			}
			else
			{
				iNeighbourCount = 8;

				piNeighbourOffsets[0] = -fio.x-1;
				piNeighbourOffsets[1] = -fio.x;
				piNeighbourOffsets[2] = -fio.x+1;
				piNeighbourOffsets[3] = -1;
				piNeighbourOffsets[4] =  1;
				piNeighbourOffsets[5] = fio.x-1;
				piNeighbourOffsets[6] = fio.x;
				piNeighbourOffsets[7] = fio.x+1;
			}
		}
		else
		{
			if( bSymetric )
			{
				iNeighbourCount = 2; //4 / 2;

				piNeighbourOffsets[0] =  1;
				piNeighbourOffsets[1] = fio.x;
			}
			else
			{
				iNeighbourCount = 4;

				piNeighbourOffsets[0] = -fio.x;
				piNeighbourOffsets[1] = -1;
				piNeighbourOffsets[2] =  1;
				piNeighbourOffsets[3] = fio.x;
			}
		}
	}

	return 1;
}


void
fioIndexToCoord(
				FEATUREIO &fio,
				const int &iIndex,
				int &x,
				int &y,
				int &z
				)
{
	x = iIndex % fio.x;
	y = (iIndex / fio.x) % fio.y;
	z = (iIndex / (fio.x*fio.y));// % fio.z; // No need to mod
}

void
fioCoordToIndex(
				FEATUREIO &fio,
				int &iIndex,
				const int &x,
				const int &y,
				const int &z
				)
{
	iIndex = x + fio.x*y + fio.x*fio.y*z;
}


int
fioGenerateHarrFilters(
					   const int &iXLevels,
					   const int &iYLevels,
					   const int &iZLevels,
					   FEATUREIO fioSize,
					   FEATUREIO *& pfioArray
				)
{
	int iFilterSize = (fioSize.x*fioSize.y*fioSize.z);
	int iArraySize = iXLevels*iYLevels*iZLevels;

	// Allocate array

	pfioArray = new FEATUREIO[iArraySize];
	if( !pfioArray )
	{
		pfioArray = 0;
		return 0;
	}

	// Allocate filters

	int i;
	for( int i = 0; i < iArraySize; i++ )
	{
		pfioArray[i] = fioSize;
		if( !fioAllocate( pfioArray[i] ) )
		{
			for( --i; i >= 0; i-- )
			{
				fioDelete( pfioArray[i] );
			}
			pfioArray = 0;
			return 0;
		}
	}

	// Factors

	int iXMult = fioSize.x / iXLevels;
	int iYMult = fioSize.y / iYLevels;
	int iZMult = fioSize.z / iZLevels;

	// Generate filters. The filter array is indexed by x, y, z

	fioSet( pfioArray[0], 1.0f );

	for( int i = 1; i < iArraySize; i++ )
	{
		// Hash i into the current filter index

		int iXFilt = i % iXLevels;
		int iYFilt = (i / iXLevels) % iYLevels;
		int iZFilt = (i / (iXLevels*iYLevels));// % iZLevels; // No need to mod

		FEATUREIO &fio = pfioArray[i];

		int bCurrZone = 0;

		for( int z = 0; z < fio.z; z++ )
		{
			int iZIndex  =  z*fio.y*fio.x*fio.iFeaturesPerVector;

			int bZZone = (z*(1<<iZFilt) / fio.z) & 1;

			for( int y = 0; y < fio.y; y++ )
			{
				int iYIndex  = iZIndex + y*fio.x*fio.iFeaturesPerVector;

				int bYZone = (y*(1<<iYFilt) / fio.y) & 1;

				for( int x = 0; x < fio.x; x++ )
				{
					int iXIndex  = iYIndex + x*fio.iFeaturesPerVector;

					int bXZone = ((x*(1<<iXFilt)) / fio.x) & 1;

					bCurrZone = bZZone + bYZone + bXZone;

					fio.pfVectors[iXIndex] = ((bCurrZone & 1) == 1) ? -1.0f : 1.0f;
				}
			}
		}
	}

	return 1;
}


int
fioGenerateEdgeImages3D(
					  FEATUREIO &fioImg,
					  FEATUREIO &fioDx,
					  FEATUREIO &fioDy,
					  FEATUREIO &fioDz
				)
{
	fioSet( fioDx, 0.0f );
	fioSet( fioDy, 0.0f );
	fioSet( fioDz, 0.0f );

	if(
		fioImg.x*fioImg.y*fioImg.z*fioImg.iFeaturesPerVector != fioDx.x*fioDx.y*fioDx.z*fioDx.iFeaturesPerVector
		||
		fioImg.x*fioImg.y*fioImg.z*fioImg.iFeaturesPerVector != fioDy.x*fioDy.y*fioDy.z*fioDy.iFeaturesPerVector
		||
		fioDx.x*fioDx.y*fioDx.z*fioDx.iFeaturesPerVector != fioDy.x*fioDy.y*fioDy.z*fioDy.iFeaturesPerVector
		)
	{
		return 0;
	}

	for( int z = 1; z < fioImg.z - 1; z++ )
	{
		int iZIndex  =  z*fioImg.y*fioImg.x;
		for( int y = 1; y < fioImg.y - 1; y++ )
		{
			for( int x = 1; x < fioImg.x - 1; x++ )
			{
				fioDx.pfVectors[z*fioImg.y*fioImg.x + y*fioImg.x + x] =
					fioImg.pfVectors[z*fioImg.y*fioImg.x + y*fioImg.x + (x+1)] - fioImg.pfVectors[z*fioImg.y*fioImg.x + y*fioImg.x + (x-1)];

				fioDy.pfVectors[z*fioImg.y*fioImg.x + y*fioImg.x + x] =
					fioImg.pfVectors[z*fioImg.y*fioImg.x + (y+1)*fioImg.x + x] - fioImg.pfVectors[z*fioImg.y*fioImg.x + (y-1)*fioImg.x + x];

				fioDz.pfVectors[z*fioImg.y*fioImg.x + y*fioImg.x + x] =
					fioImg.pfVectors[(z+1)*fioImg.y*fioImg.x + y*fioImg.x + x] - fioImg.pfVectors[(z-1)*fioImg.y*fioImg.x + y*fioImg.x + x];
			}
		}
	}
	return 1;
}

int
fioGenerateEdgeImages2D(
					  FEATUREIO &fioImg,
					  FEATUREIO &fioDx,
					  FEATUREIO &fioDy
				)
{
	fioSet( fioDx, 0.0f );
	fioSet( fioDy, 0.0f );

	if(
		fioImg.x*fioImg.y*fioImg.z*fioImg.iFeaturesPerVector != fioDx.x*fioDx.y*fioDx.z*fioDx.iFeaturesPerVector
		||
		fioImg.x*fioImg.y*fioImg.z*fioImg.iFeaturesPerVector != fioDy.x*fioDy.y*fioDy.z*fioDy.iFeaturesPerVector
		||
		fioDx.x*fioDx.y*fioDx.z*fioDx.iFeaturesPerVector != fioDy.x*fioDy.y*fioDy.z*fioDy.iFeaturesPerVector
		)
	{
		return 0;
	}

	for( int y = 1; y < fioImg.y - 1; y++ )
	{
		for( int x = 1; x < fioImg.x - 1; x++ )
		{
			fioDx.pfVectors[y*fioImg.x + x] = fioImg.pfVectors[y*fioImg.x + (x+1)] - fioImg.pfVectors[y*fioImg.x + (x-1)];
			fioDy.pfVectors[y*fioImg.x + x] = fioImg.pfVectors[(y+1)*fioImg.x + x] - fioImg.pfVectors[(y-1)*fioImg.x + x];
		}
	}
	return 1;
}

int
fioGenerateOrientationImages2D(
							 FEATUREIO &fioImg,
							 FEATUREIO &fioMag,
							 FEATUREIO &fioOri
							 )
{
	fioSet( fioMag, 0.0f );
	fioSet( fioOri, 0.0f );

	if(
		fioImg.x*fioImg.y*fioImg.z*fioImg.iFeaturesPerVector != fioMag.x*fioMag.y*fioMag.z*fioMag.iFeaturesPerVector
		||
		fioImg.x*fioImg.y*fioImg.z*fioImg.iFeaturesPerVector != fioOri.x*fioOri.y*fioOri.z*fioOri.iFeaturesPerVector
		||
		fioMag.x*fioMag.y*fioMag.z*fioMag.iFeaturesPerVector != fioOri.x*fioOri.y*fioOri.z*fioOri.iFeaturesPerVector
		)
	{
		return 0;
	}

	for( int y = 1; y < fioImg.y - 1; y++ )
	{
		for( int x = 1; x < fioImg.x - 1; x++ )
		{
			float dx = fioImg.pfVectors[y*fioImg.x + (x+1)] - fioImg.pfVectors[y*fioImg.x + (x-1)];
			float dy = fioImg.pfVectors[(y+1)*fioImg.x + x] - fioImg.pfVectors[(y-1)*fioImg.x + x];

			float fMag = sqrt( dx*dx + dy*dy );
			float fOri = atan2( dy, dx ) + PI;

			fioMag.pfVectors[y*fioImg.x + x] = fMag;

			if( fOri < 0 )
			{
				fioOri.pfVectors[y*fioImg.x + x] = PI - fOri;
			}
			else
			{
				fioOri.pfVectors[y*fioImg.x + x] = fOri;
			}
			//fioOri.pfVectors[y*fioImg.x + x] = fOri;
		}
	}
	return 1;
}

int
fioFindExtrema2D(
			   FEATUREIO &fioImg,
			   FIND_LOCATION_FUNCTION func,
			   void *pData
			   )
{
	for( int y = 1; y < fioImg.y - 1; y++ )
	{
		for( int x = 1; x < fioImg.x - 1; x++ )
		{
			int bMaximum = 1;
			int bMinimum = 1;

			float fCenterValue = fioImg.pfVectors[y*fioImg.x + x];

			for( int yy = -1; yy <= 1; yy++ )
			{
				for( int xx = -1; xx <= 1; xx++ )
				{
					if( !(yy == 0 && xx == 0) )
					{
						float fNeighbourValue = fioImg.pfVectors[(y+yy)*fioImg.x + (x+xx)];

						bMaximum = bMaximum && ( fCenterValue > fNeighbourValue );
						bMinimum = bMinimum && ( fCenterValue < fNeighbourValue );
					}
				}
			}

			if( bMaximum )
			{
				func( x, y, 1, 1, 1, fCenterValue, pData );
			}

			if( bMinimum )
			{
				func( x, y, 1, 1, 0, fCenterValue, pData );
			}
		}
	}

	return 1;
}

int
fioDoubleSize(
		   FEATUREIO &fio
		   )
{
	if( fio.iFeaturesPerVector > 1 )
	{
		printf( "Doubling not implemeneted for multidimensional images.\n" );
		assert( 0 );
		return 0;
	}
	FEATUREIO fioDouble;
	fioDouble = fio;
	if( fioDouble.x > 1 ) fioDouble.x *= 2;
	if( fioDouble.y > 1 ) fioDouble.y *= 2;
	if( fioDouble.z > 1 ) fioDouble.z *= 2;
	if( !fioAllocate( fioDouble ) )
	{
		return 0;
	}

	// This should work for 2D or 3D images

	for( int z = 0; z < fio.z; z++ )
	{
		for( int y = 0; y < fio.y; y++ )
		{
			for( int x = 0; x < fio.x; x++ )
			{
				// Get data from low-res image
				// If we get to the edge of the image, just copy values
				float pfLowRes[2][2][2]; // zz/yy/xx
				for( int zz = 0; zz <= 1; zz++ )
				{
					int dz = zz; if( z + zz >= fio.z ) dz = 0;
					for( int yy = 0; yy <= 1; yy++ )
					{
						int dy = yy; if( y + yy >= fio.y ) dy = 0;
						for( int xx = 0; xx <= 1; xx++ )
						{
							int dx = xx; if( x + xx >= fio.x ) dx = 0;
							pfLowRes[zz][yy][xx] = fioGetPixel( fio, x+dx, y+dy, z+dz );
						}
					}
				}

				if( pfLowRes[0][0][0] != 0 )
				{
					pfLowRes[0][0][0] = pfLowRes[0][0][0];
				}

				// Create high-res image
				float pfHighRes[2][2][2]; // zz/yy/xx
				pfHighRes[0][0][0] = pfLowRes[0][0][0];

				pfHighRes[1][0][0] = 0.5f*(pfLowRes[0][0][0] + pfLowRes[1][0][0]);
				pfHighRes[0][1][0] = 0.5f*(pfLowRes[0][0][0] + pfLowRes[0][1][0]);
				pfHighRes[0][0][1] = 0.5f*(pfLowRes[0][0][0] + pfLowRes[0][0][1]);

				pfHighRes[1][1][0] = 0.25f*(pfLowRes[0][0][0] + pfLowRes[1][0][0] + pfLowRes[0][1][0] + pfLowRes[1][1][0]);
				pfHighRes[0][1][1] = 0.25f*(pfLowRes[0][0][0] + pfLowRes[0][1][0] + pfLowRes[0][0][1] + pfLowRes[0][1][1]);
				pfHighRes[1][0][1] = 0.25f*(pfLowRes[0][0][0] + pfLowRes[1][0][0] + pfLowRes[0][0][1] + pfLowRes[1][0][1]);

				pfHighRes[1][1][1] = 0.125f*(pfLowRes[0][0][0]
								           + pfLowRes[0][0][1]
								           + pfLowRes[0][1][0]
								           + pfLowRes[0][1][1]
								           + pfLowRes[1][0][0]
								           + pfLowRes[1][0][1]
								           + pfLowRes[1][1][0]
								           + pfLowRes[1][1][1]);

				// Print back into image
				for( int zz = 0; zz <= 1; zz++ )
				{
					int dz = zz; if( 2*z + zz >= fioDouble.z ) dz = 0;
					for( int yy = 0; yy <= 1; yy++ )
					{
						int dy = yy; if( 2*y + yy >= fioDouble.y ) dy = 0;
						for( int xx = 0; xx <= 1; xx++ )
						{
							int dx = xx; if( 2*x + xx >= fioDouble.x ) dx = 0;
							float *pfVoxel = fioGetVector( fioDouble, 2*x+dx, 2*y+dy, 2*z+dz );
							*pfVoxel = pfHighRes[dz][dy][dx];
						}
					}
				}
			}
		}
	}

	// Delete old fio, set values to new fio
	fioDelete( fio );
	fio = fioDouble;

	return 1;
}

int
fioCreateJointImageInterleaved(
		   FEATUREIO &fioImg1,
		   FEATUREIO &fioImg2,
		   FEATUREIO &fioJoint
		   )
{
	assert( fioImg1.x == fioImg2.x );
	assert( fioImg1.y == fioImg2.y );
	assert( fioImg1.z == fioImg2.z );
	assert( fioImg1.t == fioImg2.t );
	fioJoint = fioImg1;
	fioJoint.iFeaturesPerVector = fioImg1.iFeaturesPerVector*fioImg2.iFeaturesPerVector;

	int iReturn = fioAllocate( fioJoint );
	if( iReturn == 0 )
	{
		// Could not allocate
		return 0;
	}

	// Init joint, cross product
	for( int z = 0; z < fioJoint.z; z++ )
	{
		for( int y = 0; y < fioJoint.y; y++ )
		{
			for( int x = 0; x < fioJoint.x; x++ )
			{
				float *pfProbVec1  = fioGetVector( fioImg1, x, y, z );
				float *pfProbVec2  = fioGetVector( fioImg2, x, y, z );
				float *pfProbVec = fioGetVector( fioJoint, x, y, z );

				for( int iF1 = 0; iF1 < fioImg1.iFeaturesPerVector; iF1++ )
				{
					for( int iF2 = 0; iF2 < fioImg2.iFeaturesPerVector; iF2++ )
					{
						pfProbVec[iF1*fioImg1.iFeaturesPerVector + iF2] = pfProbVec1[iF1]*pfProbVec2[iF2];
					}
				}
			}
		}
	}

	return 1;
}

int
fioInitJointImageArray(
		   FEATUREIO &fioImg1,
		   FEATUREIO &fioImg2,
		   FEATUREIO *fioJointArray // Array of FEATUREIO
		   )
{
	assert( fioImg1.x == fioImg2.x );
	assert( fioImg1.y == fioImg2.y );
	assert( fioImg1.z == fioImg2.z );
	assert( fioImg1.t == fioImg2.t );

	// Don't allocate all arrays
	//for( int iF1 = 0; iF1 < fioImg1.iFeaturesPerVector; iF1++ )
	//{
	//	for( int iF2 = 0; iF2 < fioImg2.iFeaturesPerVector; iF2++ )
	//	{
	//		fioJointArray[iF1*fioImg1.iFeaturesPerVector + iF2] = fioImg1;
	//		fioJointArray[iF1*fioImg1.iFeaturesPerVector + iF2].iFeaturesPerVector = 1;
	//		int iReturn = fioAllocate( fioJointArray[iF1*fioImg1.iFeaturesPerVector + iF2] );
	//		if( iReturn == 0 )
	//		{
	//			// Could not allocate
	//			return 0;
	//		}
	//	}
	//}

	// Init joint, cross product
	for( int z = 0; z < fioImg1.z; z++ )
	{
		for( int y = 0; y < fioImg1.y; y++ )
		{
			for( int x = 0; x < fioImg1.x; x++ )
			{
				float *pfProbVec1  = fioGetVector( fioImg1, x, y, z );
				float *pfProbVec2  = fioGetVector( fioImg2, x, y, z );

				for( int iF1 = 0; iF1 < fioImg1.iFeaturesPerVector; iF1++ )
				{
					for( int iF2 = 0; iF2 < fioImg2.iFeaturesPerVector; iF2++ )
					{
						float *pfProbJoint = fioGetVector( fioJointArray[iF1*fioImg1.iFeaturesPerVector + iF2], x, y, z );
						*pfProbJoint = pfProbVec1[iF1]*pfProbVec2[iF2];
					}
				}
			}
		}
	}

	return 1;
}

int
fioInitJointImageArray(
		   FEATUREIO *fioImg1,
		   FEATUREIO *fioImg2,
		   int iCount1,
		   int iCount2,
		   FEATUREIO *fioJointArray // Array of FEATUREIO
		   )
{
	assert( fioImg1[0].x == fioImg2[0].x );
	assert( fioImg1[0].y == fioImg2[0].y );
	assert( fioImg1[0].z == fioImg2[0].z );
	assert( fioImg1[0].t == fioImg2[0].t );

	// Init joint, cross product
	for( int z = 0; z < fioJointArray[0].z; z++ )
	{
		for( int y = 0; y < fioJointArray[0].y; y++ )
		{
			for( int x = 0; x < fioJointArray[0].x; x++ )
			{
				for( int iF1 = 0; iF1 < iCount1; iF1++ )
				{
					for( int iF2 = 0; iF2 < iCount2; iF2++ )
					{
						float *pfProbVec1  = fioGetVector( fioImg1[iF1], x, y, z );
						float *pfProbVec2  = fioGetVector( fioImg2[iF2], x, y, z );
						float *pfProbJoint = fioGetVector( fioJointArray[iF1*iCount1 + iF2], x, y, z );
						*pfProbJoint = pfProbVec1[0]*pfProbVec2[0];
					}
				}
			}
		}
	}

	return 1;
}

int
fioNormalizeJointImageArray(
		   int iCount1,
		   int iCount2,
		   FEATUREIO *fioJointArray // Array of FEATUREIO
							)
{
	float fSum = 0;
	for( int z = 0; z < fioJointArray[0].z; z++ )
	{
		for( int y = 0; y < fioJointArray[0].y; y++ )
		{
			for( int x = 0; x < fioJointArray[0].x; x++ )
			{
				fSum = 0;
				for( int iF1 = 0; iF1 < iCount1; iF1++ )
				{
					for( int iF2 = 0; iF2 < iCount2; iF2++ )
					{
						float *pfProbJoint = fioGetVector( fioJointArray[iF1*iCount1 + iF2], x, y, z );
						fSum += *pfProbJoint;
					}
				}
				float fSumDiv = 1.0f/fSum;
				fSum = 0;
				for( int iF1 = 0; iF1 < iCount1; iF1++ )
				{
					for( int iF2 = 0; iF2 < iCount2; iF2++ )
					{
						float *pfProbJoint = fioGetVector( fioJointArray[iF1*iCount1 + iF2], x, y, z );
						float fProbNorm = (*pfProbJoint)*fSumDiv;
						*pfProbJoint = fProbNorm;
						fSum += fProbNorm;
					}
				}
			}
		}
	}
	return 0;
}

void
fioCalculateSVD(
				FEATUREIO &fioIn,
				FEATUREIO &fioPCs,
				float *pfEigenValues
			   )
{
	assert( fioIn.t > 1 );
	assert( fioIn.iFeaturesPerVector == 1 );
	assert( fioPCs.iFeaturesPerVector == 1 );

	int iImgSize = fioIn.x*fioIn.y*fioIn.z;
	double *pdCov = new double[iImgSize*iImgSize];
	double *pdEig = new double[iImgSize];
	double *pdVec = new double[iImgSize*iImgSize];

	double **pdCovPt = new double*[iImgSize];
	double **pdVecPt = new double*[iImgSize];
	for( int i = 0; i < iImgSize; i++ )
	{
		pdCovPt[i] = &pdCov[i*iImgSize];
		pdVecPt[i] = &pdVec[i*iImgSize];
		for( int j = 0; j < iImgSize; j++ )
		{
			pdCovPt[i][j] = 0;
			pdVecPt[i][j] = 0;
		}
	}

	// Compute mean
	double *pdMean = new double[iImgSize];
	for( int i =0; i < iImgSize; i++ )
	{
		pdMean[i] = 0;
	}
	for( int iImg = 0; iImg < fioIn.t; iImg++ )
	{
		float *pfVec = fioGetVector( fioIn, 0, 0, 0, iImg );
		for( int i = 0; i < iImgSize; i++ )
		{
			pdMean[i] += pfVec[i];
		}
	}
	for( int i =0; i < iImgSize; i++ )
	{
		pdMean[i] /= (float)fioIn.t;
	}

	// Accumulate covariance matrix
	int iSamples = 0;
	for( int iImg = 0; iImg < fioIn.t; iImg++ )
	{
		float *pfVec = fioGetVector( fioIn, 0, 0, 0, iImg );

		for( int i =0; i < iImgSize; i++ )
		{
			pfVec[i] -= pdMean[i];
		}
		for( int i =0; i < iImgSize; i++ )
		{
			for( int j=0; j < iImgSize; j++ )
			{
				pdCovPt[i][j] += pfVec[i]*pfVec[j];
				//pdCovPt[i][j] += (feat.data_zyx[0][0][i]-pdMean[i])*(feat.data_zyx[0][0][j]-pdMean[j]);
			}
		}
		if( iImg % 10 == 0 )
		{
			printf( "Feature: %d of %d\n", iImg, fioIn.t );
		}
		iSamples++;
	}

	// Normalize number of samples
	for( int i =0; i < iImgSize; i++ )
	{
		for( int j=0; j < iImgSize; j++ )
		{
			pdCovPt[i][j] /= (double)iSamples;
		}
	}

	// Determine principal components
	// svdcmp_iterations( pdCovPt, iImgSize, iImgSize, pdEig, pdVecPt, 1000 );
	// reorder_descending( iImgSize, iImgSize, pdEig, pdVecPt );

	// Save 1st principal components
	for( int iImg = 0; iImg < fioPCs.t; iImg++ )
	{
		float *pfVec = fioGetVector( fioPCs, 0, 0, 0, iImg );

		for( int i =0; i < iImgSize; i++ )
		{
			pfVec[i] = pdVecPt[i][iImg];
		}

		// Save eigen value
		pfEigenValues[iImg] = pdEig[iImg];
	}

	delete [] pdMean;
	delete [] pdCov;
	delete [] pdEig;
	delete [] pdVec;
	delete [] pdCovPt;
	delete [] pdVecPt;
}

int
fioTranslate(
			 FEATUREIO &fioDst,
			FEATUREIO &fioSrc,
			int dx, int dy, int dz
				)
{
	assert( fioDst.iFeaturesPerVector == fioSrc.iFeaturesPerVector );
	fioSet( fioDst, 0 );
	for( int z = 0; z < fioDst.z; z++ )
	{
		for( int y = 0; y < fioDst.y; y++ )
		{
			for( int x = 0; x < fioDst.x; x++ )
			{
				int iSrcx = x - dx;
				int iSrcy = y - dy;
				int iSrcz = z - dz;
				if( iSrcx > 0 && iSrcx < fioSrc.x &&
					iSrcy > 0 && iSrcy < fioSrc.y &&
					iSrcz > 0 && iSrcz < fioSrc.z )
				{
					float *pfVecDst = fioGetVector( fioDst, x, y, z );
					float *pfVecSrc = fioGetVector( fioSrc, iSrcx, iSrcy, iSrcz );
					for( int k = 0; k < fioSrc.iFeaturesPerVector; k++ )
					{
						pfVecDst[k] = pfVecSrc[k];
					}
				}
			}
		}
	}
	return 1;
}

int
fioPixelCount(
			 FEATUREIO &fio
			  )
{
	return fio.x*fio.y*fio.z*fio.t;
}

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
				 )
{
	float dx = x2-x1;
	float dy = y2-y1;
	float dz = z2-z1;
	float fDistSqr = dx*dx + dy*dy + dz*dz;
	if( fDistSqr <= 0 )
	{
		return 0;
	}
	float fDist = sqrt( fDistSqr );
	float fDistInc = fDist / (float)iCount;

	// Make
	dx = dx*fDistInc /fDist;
	dy = dy*fDistInc /fDist;
	dz = dz*fDistInc /fDist;
	for( int i =0; i < iCount; i++ )
	{
		float fPixel;
		if( fio.z > 1 )
		{
			fPixel = fioGetPixelTrilinearInterp( fio,
				x1+dx*i+0.5,
				y1+dy*i+0.5,
				z1+dz*i+0.5 );
		}
		else
		{
			fPixel = fioGetPixelBilinearInterp( fio,
				x1+dx*i+0.5,
				y1+dy*i+0.5 );
		}
		pfLine[i] = fPixel;
	}
	return 1;
}

//FEATUREIO *
//fioGeneratePyramid(
//				   FEATUREIO &fioTop,
//				   int &iLevels
//				   )
//{
//	int iMinDim = fioTop.x;
//	if( fioTop.y > 1 && fioTop.y > iMinDim )
//	{
//		iMinDim = fioTop.y;
//	}
//	if( fioTop.z > 1 && fioTop.z > iMinDim )
//	{
//		iMinDim = fioTop.z;
//	}
//
//	for( iLevels = 1; 0 < (iMinDim / iLevels); iLevels++ )
//	{
//	}
//
//	FEATUREIO *pfio = new FEATUREIO[iLevels];
//}

template<class TYPE>
static int
_correlate_diff_two(
		   TYPE *ptTemplate, int iTemplateSize,
		   TYPE *ptSignal, int iSignalInc,
		   float *ptResult, int iResultSize
		   )
{
	for( int i = 0; i < iResultSize; i++ )
	{
		float dSumXY = 0;
		for( int j = 0; j < iTemplateSize; j++ )
		{
			float fDiff = ptTemplate[ j ] - ptSignal[ -i*iSignalInc + j ];
			dSumXY += fabs( fDiff );
		}

		float dSumXY2 = 0;
		for( int j = 0; j < iTemplateSize; j++ )
		{
			float fDiff = ptTemplate[ j ] - ptSignal[ i*iSignalInc + j ];
			dSumXY2 += fabs( fDiff );
		}
		ptResult[i] += dSumXY + dSumXY2;
	}

	return 1;
}


static bool
_sortIndexFloatAscendingValue(
		   const pair<int,float> &p1,
		   const pair<int,float> &p2
		   )
{
	return p1.second < p2.second;
}

static bool
_sortIndexFloatAscendingIndex(
		  const  pair<int,float> &p1,
		  const  pair<int,float> &p2
		   )
{
	return p1.first < p2.first;
}


//
// get_image_dimensions()
//
// Robust dimension determination.
//
template<class TYPE>
static int
get_image_dimensions(
		   TYPE *ptSignal,
		   int iSignalSize,
		   int *piDims
		   )
{
	int iCurrDim = 0;
	int iResultSamples = 2048;
	float *pfResult = new float[iResultSamples];
	int iDimInc = 1;
	int iTempSize = 5;
	vector< pair<int,float> > vecIndexValue;

	double dDiffAvg = 0;
	for( int i = 0; i < iSignalSize-1; i++ )
	{
		// Check for NAN or INF, IEEE 32-bit float format
		//if( ptSignal[i] != ptSignal[i] || (ptSignal[i] == ptSignal[i] + ptSignal[i] && ptSignal[i] != 0) )
		if( ((int)(ptSignal[i]) & 0x7F800000) == 0x7F800000 )
		{
			//return -1;
		}
		dDiffAvg += fabs( (float)(ptSignal[i] - ptSignal[i+1]) );
	}
	dDiffAvg /= (double)(iSignalSize-1);

	int iTries = 0;
	int iMaxTries = 20;

	if( iSignalSize < iResultSamples )
	{
		return -1;
	}

	while( iDimInc < iSignalSize && iTries < iMaxTries )
	{
		memset( pfResult, 0, sizeof(float)*iResultSamples );
		for( int i = 0; i < 300; i++ )
		{
			// Find random template
			double dRand;
			int iTempStart = iSignalSize;
			while( iTempStart + iResultSamples*iDimInc + iTempSize >= iSignalSize
				||
				iTempStart - iResultSamples*iDimInc - iTempSize < 0 )
			{
				dRand = rand()/(float)RAND_MAX;
				iTempStart = dRand*iSignalSize;

				//
				// Does not work - better to start at a random location than  at an edge
				//while( iTempStart + iResultSamples*iDimInc + iTempSize < iSignalSize )
				//{
				//	//float fDiff = fabs( (float)(ptSignal[iTempStart/2] - ptSignal[iTempStart/2+1]) );
				//	int bEqual = (ptSignal[iTempStart/2] == ptSignal[iTempStart/2+1]);
				//	//if( fDiff > dDiffAvg )
				//	if( !bEqual )
				//	{
				//		break;
				//	}
				//	iTempStart++;
				//}
			}

			_correlate_diff_two(
			//correlate_diff_(
				ptSignal + iTempStart, iTempSize,
				ptSignal + iTempStart, iDimInc ,
				pfResult, iResultSamples
			   );
		}

		//FILE *outfile = fopen( "data-res.txt", "wt" );
		//for( int i = 0; i < iResultSamples; i++ )
		//	fprintf( outfile, "%d\t%f\n", i, pfResult[i] );
		//fclose( outfile );

		for( int i = 1; i < iResultSamples/2; i++ )
		{
			float fValue = 0;
			int iCount = 0;
			int iTotalSamples = iResultSamples/i;
			int iIncrementFactor = iTotalSamples / 27;
			if( iIncrementFactor < 1 )
			{
				iIncrementFactor = 1;
			}
			for( int j = i; j < iResultSamples; j += i*iIncrementFactor )
			{
				fValue += pfResult[j];
				iCount++;
			}
			fValue = fValue / (float)iCount;
			pfResult[i] = fValue;
		}

		//outfile = fopen( "data-res-fold-sumavg2.txt", "wt" );
		//for( int i = 0; i < iResultSamples; i++ )
		//	fprintf( outfile, "%d\t%f\n", i, pfResult[i] );
		//fclose( outfile );

		// Save peak list
		vecIndexValue.clear();
		for( int i = 1; i < iResultSamples/2-1; i++ )
		{
			if( pfResult[i] < pfResult[i-1] && pfResult[i] < pfResult[i+1] )
			{
				vecIndexValue.push_back( pair<int,float>(i,pfResult[i]) );
			}
		}
		if( vecIndexValue.size() <= 1 )
		{
			// Error
			return -1;
		}

		// Sort peaks by value
		sort( vecIndexValue.begin(), vecIndexValue.end(), _sortIndexFloatAscendingValue);

		// Identfy maximum derivative
		int iMaxDiffIndex = 1;
		float fMaxDiffValue = fabs( vecIndexValue[0].second - vecIndexValue[1].second );
		for( int i = 0; i < vecIndexValue.size()-1; i++ )
		{
			float fDiffValue = fabs( vecIndexValue[i].second - vecIndexValue[i+1].second );
			if( fDiffValue > fMaxDiffValue )
			{
				iMaxDiffIndex = i+1;
				fMaxDiffValue = fDiffValue;
			}
		}
		// Remove all
		vecIndexValue.erase( vecIndexValue.begin() + iMaxDiffIndex, vecIndexValue.end() );

		// Sort peaks of frequency
		sort( vecIndexValue.begin(), vecIndexValue.end(), _sortIndexFloatAscendingIndex );

		// Pick minimum frequency
		int iMinIndex = vecIndexValue[0].first;

		piDims[iCurrDim] = iMinIndex;
		iCurrDim++;
		iDimInc *= iMinIndex;
		if( iSignalSize / iDimInc < iResultSamples/2 )
		{
			// Less than half samples in remaining dimension
			if( iSignalSize % iDimInc != 0 )
			{
				// Should be even multiple - restart
				iCurrDim = 0;
				iDimInc = 1;
				iTries++;
			}
			else
			{
				// Success!
				//printf( "Tries: %d\n", iTries );
				piDims[iCurrDim] = iSignalSize / iDimInc;
				delete [] pfResult;
				return 1;
			}
		}
	}

	delete [] pfResult;
	//printf( "Tries: %d\n", iTries );
	return -1;
}

int
fioReadRaw(
				 FEATUREIO &fio,
				 char *pcFileName
				 )
{
	FILE *infile = fopen( pcFileName, "rb" );
	if( !infile )
	{
		return -1;
	}
	fseek( infile, 0, SEEK_END );
	int iSizeBytes = ftell(infile);
	fseek( infile, 0, SEEK_SET );
	unsigned char *pucData = new unsigned char[iSizeBytes];
	if( !pucData )
	{
		fclose( infile );
		return -2;
	}
	int iResult = fread( pucData, 1, iSizeBytes, infile );
	fclose( infile );
	if( iResult != iSizeBytes )
	{
		delete [] pucData;
		return -3;
	}

	int piDims[5] = {1,1,1,1,1};
	iResult = get_image_dimensions( (float*)pucData, iSizeBytes/sizeof(float), piDims );
	//iResult = get_image_dimensions( pucData, iSizeBytes, piDims );
	if( iResult < 0 )
	{
		memset( &fio, 0, sizeof(fio) );
		delete [] pucData;
	}
	else
	{
		fio.pfVectors = (float*)pucData;
		fio.iFeaturesPerVector = 1;
		fio.x = fio.y = fio.z = fio.t = 1;
		fio.x = piDims[0];
		fio.y = piDims[1];
		fio.z = piDims[2];
		fio.t = piDims[3];
		fio.pfMeans = new float[1];
		fio.pfVarrs = new float[1];
	}

	return iResult;
}

int
fioFlipAxisY(
		FEATUREIO &fio
			 )
{
	for( int z = 0; z < fio.z; z++ )
	{
		for( int y = 0; y < fio.y/2; y++ )
		{
			for( int x = 0; x < fio.x; x++ )
			{
				float *pfVec1 = fioGetVector( fio, x, y, z );
				float *pfVec2 = fioGetVector( fio, x, fio.y - 1 - y, z );
				for( int k = 0; k < fio.iFeaturesPerVector; k++ )
				{
					float fTemp = pfVec1[k];
					pfVec1[k] = pfVec2[k];
					pfVec2[k] = fTemp;
				}
			}
		}
	}
	return 1;
}
int
fioFlipAxisX(
		FEATUREIO &fio
			 )
{
	for( int z = 0; z < fio.z; z++ )
	{
		for( int y = 0; y < fio.y; y++ )
		{
			for( int x = 0; x < fio.x/2; x++ )
			{
				float *pfVec1 = fioGetVector( fio, x, y, z );
				float *pfVec2 = fioGetVector( fio, fio.x - 1 - x, y, z );
				for( int k = 0; k < fio.iFeaturesPerVector; k++ )
				{
					float fTemp = pfVec1[k];
					pfVec1[k] = pfVec2[k];
					pfVec2[k] = fTemp;
				}
			}
		}
	}
	return 1;
}
