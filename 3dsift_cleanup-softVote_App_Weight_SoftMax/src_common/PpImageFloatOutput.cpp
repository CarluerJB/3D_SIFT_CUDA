
#include <assert.h>
#include <stdio.h>
#include "PpImageFloatOutput.h"

#define MIN_GREYSCALE	0
#define MAX_GREYSCALE	255

// Match structure

typedef struct
{
	int				iRow;
	int				iCol;
	float			fResult;
	unsigned long	ulResult;
	int				iResult;
} MATCH_RESULT;


//
// min_max_float()
//
// ppImg is an image of interleaved float feature vectors. The min max
// search is performed by default on the first feature in each vector.
//
void
min_max_float(
	const PpImage &ppImg,
	MATCH_RESULT &mrMin,
	MATCH_RESULT &mrMax,
	int iFeature = 0
		 )
{
	int iFeaturesPerVector = (ppImg.BitsPerPixel()/8)/sizeof(float);

	float *pfImgRowI = (float*)ppImg.ImageRow(0);
	mrMax.fResult = pfImgRowI[0];
	mrMin.fResult = pfImgRowI[0];
	mrMax.iRow = 0;
	mrMax.iCol = 0;
	mrMin.iRow = 0;
	mrMin.iCol = 0;
	for( int i = 0; i < ppImg.Rows(); i++ )
	{
		float *pfImgRow = (float*)ppImg.ImageRow( i );

		for( int j = 0; j < ppImg.Cols(); j++ )
		{
			if( pfImgRow[j*iFeaturesPerVector + iFeature] > mrMax.fResult )
			{
				mrMax.fResult = pfImgRow[j*iFeaturesPerVector + iFeature];
				mrMax.iRow = i;
				mrMax.iCol = j;
			}
			if( pfImgRow[j*iFeaturesPerVector + iFeature] < mrMin.fResult )
			{
				mrMin.fResult = pfImgRow[j*iFeaturesPerVector + iFeature];
				mrMin.iRow = i;
				mrMin.iCol = j;
			}
		}
	}
}

void
min_max_int(
	const PpImage &ppImg,	// Image of unsigned long
	MATCH_RESULT &mrMin,
	MATCH_RESULT &mrMax
		 )
{

	mrMax.iResult = 0x80000000;
	mrMin.iResult = 0x7FFFFFFF;

	for( int i = 0; i < ppImg.Rows(); i++ )
	{
		int *piImgRow = (int*)ppImg.ImageRow( i );

		for( int j = 0; j < ppImg.Cols(); j++ )
		{
			if( piImgRow[j] > mrMax.iResult )
			{
				mrMax.iResult = piImgRow[j];
				mrMax.iRow = i;
				mrMax.iCol = j;
			}
			if( piImgRow[j] < mrMin.iResult )
			{
				mrMin.iResult = piImgRow[j];
				mrMin.iRow = i;
				mrMin.iCol = j;
			}
		}
	}
}

//
// output_text()
//
// Output image as a text file.
//
void
output_float_to_text(
	const PpImage &ppImgFloat,
	char *pcFName,
	int iFeature
	)
{
	FILE *outfile = fopen( pcFName, "wt" );
	if( !outfile )
	{
		return;
	}
	int iFeaturesPerVector = (ppImgFloat.BitsPerPixel()/8)/sizeof(float);
	for( int i = 0; i < ppImgFloat.Rows(); i++ )
	{
		float *pfImgRow = (float*)ppImgFloat.ImageRow( i );

		for( int j = 0; j < ppImgFloat.Cols(); j++ )
		{
			fprintf( outfile, "%f\t", pfImgRow[j*iFeaturesPerVector + iFeature] );
		}
		fprintf( outfile, "\n" );
	}
	fclose( outfile );
}

//
// output_float()
//
// Outputs a float image to a character image.
//
void
output_float(
	const PpImage &ppImgFloat,
	PpImage &ppImgChar,
	int iFeature
	)
{
	if( !ppImgChar.Initialize( ppImgFloat.Rows(), ppImgFloat.Cols(), ppImgFloat.Cols(), 8 ) )
	{
		printf( "Error: output_float() cannot allocate image.\n" );
		return;
	}

	MATCH_RESULT mrMin;
	MATCH_RESULT mrMax;

	min_max_float( ppImgFloat, mrMin, mrMax, iFeature );

	int iFeaturesPerVector = (ppImgFloat.BitsPerPixel()/8)/sizeof(float);

	for( int i = 0; i < ppImgFloat.Rows(); i++ )
	{
		float *pfImgRow = (float*)ppImgFloat.ImageRow( i );
		unsigned char *pcImgOutRow = ppImgChar.ImageRow( i );

		for( int j = 0; j < ppImgFloat.Cols(); j++ )
		{
			pcImgOutRow[ j ] = (unsigned char)
				(((pfImgRow[j*iFeaturesPerVector + iFeature] - mrMin.fResult)*255.0)/(mrMax.fResult - mrMin.fResult));
		}
	}
}

void
output_float(
	const PpImage &ppImg,
	char *pcFName,
	int iFeature
	)
{
	PpImage ppImgOut;

	output_float( ppImg, ppImgOut, iFeature	);

	ppImgOut.WriteToFile( pcFName );
}

void
output_int(
	const PpImage &ppImg,
	char *pcFName
	)
{
	PpImage ppImgOut;
	ppImgOut.Initialize( ppImg.Rows(), ppImg.Cols(), ppImg.Cols(), 8 );

	MATCH_RESULT mrMin;
	MATCH_RESULT mrMax;
	
	min_max_int( ppImg, mrMin, mrMax );
	if( mrMax.iResult == mrMin.iResult )
	{
		for( int i = 0; i < ppImg.Rows(); i++ )
		{
			int *piImgRow = (int*)ppImg.ImageRow( i );
			unsigned char *pcImgOutRow = ppImgOut.ImageRow( i );

			for( int j = 0; j < ppImg.Cols(); j++ )
			{
				pcImgOutRow[ j ] = 255;
			}
		}
	}
	else
	{
		for( int i = 0; i < ppImg.Rows(); i++ )
		{
			int *piImgRow = (int*)ppImg.ImageRow( i );
			unsigned char *pcImgOutRow = ppImgOut.ImageRow( i );

			for( int j = 0; j < ppImg.Cols(); j++ )
			{
				pcImgOutRow[ j ] = (unsigned char)
					(((piImgRow[ j ] - mrMin.iResult)*255)/(mrMax.iResult - mrMin.iResult));
			}
		}
	}

	ppImgOut.WriteToFile( pcFName );
}

void
output_float_clip(
	const PpImage &ppImg,
	char *pcFName,
	const float fLower,
	const float fUpper,
	int iFeature
	)
{
	PpImage ppImgOut;
	ppImgOut.Initialize( ppImg.Rows(), ppImg.Cols(), ppImg.Cols(), 8 );

	int iFeaturesPerVector = (ppImg.BitsPerPixel()/8)/sizeof(float);

	for( int i = 0; i < ppImg.Rows(); i++ )
	{
		float *pfImgRow = (float*)ppImg.ImageRow( i );
		unsigned char *pcImgOutRow = ppImgOut.ImageRow( i );

		for( int j = 0; j < ppImg.Cols(); j++ )
		{
			float fValue;
			if( pfImgRow[j] < fLower )
			{
				fValue = fLower;
			}
			else if( pfImgRow[j] > fUpper )
			{
				fValue = fUpper;
			}
			else
			{
				fValue = pfImgRow[j];
			}

			pcImgOutRow[ j ] = (unsigned char)
				(((pfImgRow[j*iFeaturesPerVector + iFeature] - fLower)*255.0)/(fUpper - fLower));
		}
	}

	ppImgOut.WriteToFile( pcFName );
}


