
#include "LocationValue.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

static int
compare_location_value_xyz_low_high(
				   const void *pv1,
				   const void *pv2
				   )
{
	LOCATION_VALUE_XYZ *plv1 = (LOCATION_VALUE_XYZ*)pv1;
	LOCATION_VALUE_XYZ *plv2 = (LOCATION_VALUE_XYZ*)pv2;
	if( plv1->fValue < plv2->fValue )
	{
		return -1;
	}
	else if( plv1->fValue  > plv2->fValue )
	{
		return 1;
	}
	return 0;
}

static int
compare_location_value_xyz_high_low(
				   const void *pv1,
				   const void *pv2
				   )
{
	LOCATION_VALUE_XYZ *plv1 = (LOCATION_VALUE_XYZ*)pv1;
	LOCATION_VALUE_XYZ *plv2 = (LOCATION_VALUE_XYZ*)pv2;
	if( plv1->fValue < plv2->fValue )
	{
		return 1;
	}
	else if( plv1->fValue > plv2->fValue )
	{
		return -1;
	}
	return 0;
}


int
lvSortHighLow(
			  LOCATION_VALUE_XYZ_ARRAY &lva
			  )
{
	qsort( lva.plvz, lva.iCount, sizeof(LOCATION_VALUE_XYZ),
		compare_location_value_xyz_high_low );
	return 1;
}

int
lvSortLowHigh(
			  LOCATION_VALUE_XYZ_ARRAY &lva
			  )
{
	qsort( lva.plvz, lva.iCount, sizeof(LOCATION_VALUE_XYZ),
		compare_location_value_xyz_low_high );
	return 1;
}


int
lvInitNeighboursFullyConnected(
						const LOCATION_VALUE_XYZ_ARRAY &lvaVariables,
						const int iVariables,
						unsigned char *pucNeighbours
						)
{
	memset( pucNeighbours, 255, iVariables*iVariables*sizeof(unsigned char));
	return 1;
}

int
lvEnforceRangeX(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
			   int iMin,
			   int iMax
			   )
{
	for( int i = 0; i < lpaVariables.iCount; )
	{
		// Test to see if current variable is greater than
		// min dist away from all previous variables.

		LOCATION_VALUE_XYZ &lvCurr = lpaVariables.plvz[i];

		if( lvCurr.x < iMin || lvCurr.x > iMax )
		{
			// Out of range, remove
			for( int j = i + 1; j < lpaVariables.iCount; j++ )
			{
				LOCATION_VALUE_XYZ &lv1 = lpaVariables.plvz[j-1];
				LOCATION_VALUE_XYZ &lv2 = lpaVariables.plvz[j];
				lv1 = lv2;
			}
			lpaVariables.iCount--;
		}
		else
		{
			i++;
		}
	}

	return 1;
}

int
lvEnforceRangeY(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
			   int iMin,
			   int iMax
			   )
{
	for( int i = 0; i < lpaVariables.iCount; )
	{
		// Test to see if current variable is greater than
		// min dist away from all previous variables.

		LOCATION_VALUE_XYZ &lvCurr = lpaVariables.plvz[i];

		if( lvCurr.y < iMin || lvCurr.y > iMax )
		{
			// Out of range, remove
			for( int j = i + 1; j < lpaVariables.iCount; j++ )
			{
				LOCATION_VALUE_XYZ &lv1 = lpaVariables.plvz[j-1];
				LOCATION_VALUE_XYZ &lv2 = lpaVariables.plvz[j];
				lv1 = lv2;
			}
			lpaVariables.iCount--;
		}
		else
		{
			i++;
		}
	}

	return 1;
}

int
lvEnforceRangeZ(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
			   int iMin,
			   int iMax
			   )
{
	for( int i = 0; i < lpaVariables.iCount; )
	{
		// Test to see if current variable is greater than
		// min dist away from all previous variables.

		LOCATION_VALUE_XYZ &lvCurr = lpaVariables.plvz[i];

		if( lvCurr.z < iMin || lvCurr.z > iMax )
		{
			// Out of range, remove
			for( int j = i + 1; j < lpaVariables.iCount; j++ )
			{
				LOCATION_VALUE_XYZ &lv1 = lpaVariables.plvz[j-1];
				LOCATION_VALUE_XYZ &lv2 = lpaVariables.plvz[j];
				lv1 = lv2;
			}
			lpaVariables.iCount--;
		}
		else
		{
			i++;
		}
	}

	return 1;
}

int
lvDistSqr(
		  const LOCATION_VALUE_XYZ &lv1,
		  const LOCATION_VALUE_XYZ &lv2
		  )
{
	int iDistX = lv1.x - lv2.x;
	int iDistY = lv1.y - lv2.y;
	int iDistZ = lv1.z - lv2.z;
	return iDistZ*iDistZ + iDistY*iDistY + iDistX*iDistX;
}

float
lvDistSqrFloat(
		  const LOCATION_VALUE_XYZ &lv1,
		  const LOCATION_VALUE_XYZ &lv2
		  )
{
	float fDistX = (float)(lv1.x - lv2.x);
	float fDistY = (float)(lv1.y - lv2.y);
	float fDistZ = (float)(lv1.z - lv2.z);
	return fDistZ*fDistZ + fDistY*fDistY + fDistX*fDistX;
}

int
lvEnforceMinAbsValue(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
				float fAbsValue
			   )
{
	for( int i = 0; i < lpaVariables.iCount; )
	{
		// Test to see if current variable is greater than
		// min dist away from all previous variables.

		LOCATION_VALUE_XYZ &lvCurr = lpaVariables.plvz[i];

		if( fabs( lvCurr.fValue ) < fAbsValue )
		{
			// Out of range, remove
			for( int j = i + 1; j < lpaVariables.iCount; j++ )
			{
				LOCATION_VALUE_XYZ &lv1 = lpaVariables.plvz[j-1];
				LOCATION_VALUE_XYZ &lv2 = lpaVariables.plvz[j];
				lv1 = lv2;
			}
			lpaVariables.iCount--;
		}
		else
		{
			i++;
		}
	}

	return 1;
}

int
lvEnforceMinDist(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
			   int iMinDist
			   )
{
	int i, j, iCurrValidCount;

	iCurrValidCount = 1;

	int iMinDistSqr = iMinDist*iMinDist;

	for( int i = 1; i < lpaVariables.iCount; i++ )
	{
		// Test to see if current variable is greater than
		// min dist away from all previous variables.

		LOCATION_VALUE_XYZ &lvCurr = lpaVariables.plvz[i];

		int iDistSqr = iMinDistSqr + 1;

		for( int j = 0; j < iCurrValidCount && iDistSqr > iMinDistSqr; j++ )
		{
			LOCATION_VALUE_XYZ &lvPrev = lpaVariables.plvz[j];

			int iDistX = lvCurr.x - lvPrev.x;
			int iDistY = lvCurr.y - lvPrev.y;
			int iDistZ = lvCurr.z - lvPrev.z;
			iDistSqr = iDistZ*iDistZ + iDistY*iDistY + iDistX*iDistX;
		}

		if( iDistSqr >= iMinDistSqr )
		{
			// Min dist is valid, keep
			lpaVariables.plvz[iCurrValidCount] = lpaVariables.plvz[i];
			iCurrValidCount++;
		}
	}

	lpaVariables.iCount = iCurrValidCount;

	return 1;
}

int
lvWriteLocationValueXYZArray(
						   LOCATION_VALUE_XYZ_ARRAY &LPA,
						   char *pcFileName
						   )
{
	FILE *outfile = fopen( pcFileName, "wt" );
	if( !outfile )
	{
		return 0;
	}
	for( int i = 0; i < LPA.iCount; i++ )
	{
		fprintf( outfile, "%d\t%d\t%d\t%100.100f\n", LPA.plvz[i].x, LPA.plvz[i].y, LPA.plvz[i].z, LPA.plvz[i].fValue );
	}
	fclose( outfile );
	return 1;
}

int
lvReadLocationValueXYZArray(
						   LOCATION_VALUE_XYZ_ARRAY &lva,
						   char *pcFileName
					   )
{
	lva.iCount = 0;
	if( !pcFileName )
	{
	}
	FILE *infile = fopen( pcFileName, "rt" );
	if( !infile )
	{
		return 0;
	}

	int iBufferSize = 2000;
	int iMaxCount = 0;
	char *pcBuffer = new char[iBufferSize];
	if( !pcBuffer )
	{
		return 0;
	}

	while( fgets( pcBuffer, iBufferSize, infile ) )
	{
		iMaxCount++;
	}

	
	lva.plvz = new LOCATION_VALUE_XYZ[iMaxCount];
	if( !lva.plvz )
	{
		fclose( infile );
		delete [] pcBuffer;
		return 0;
	}

	lva.iCount = 0;
	fseek( infile, 0, SEEK_SET );
	while( fgets( pcBuffer, iBufferSize, infile ) )
	{
		LOCATION_VALUE_XYZ &lvCurr = lva.plvz[lva.iCount];

		float fx, fy, fz, fv;
		if( sscanf( pcBuffer, "%f\t%f\t%f\t%f\n",
			&fx, &fy, &fz, &fv ) == 4 )
		{
			lvCurr.x = (int)fx;
			lvCurr.y = (int)fy;
			lvCurr.z = (int)fz;
			lvCurr.fValue = fv;
			lva.iCount++;
		}

		//if( sscanf( pcBuffer, "%d\t%d\t%d\t%f\n",
		//	&lvCurr.x, &lvCurr.y, &lvCurr.z, &lvCurr.fValue ) == 4 )
		//{
		//	lva.iCount++;
		//}
	}

	fclose( infile );

	delete [] pcBuffer;
	return 1;
}

int
lvCopy(
	   LOCATION_VALUE_XYZ_ARRAY			&lvaDst,
	   const LOCATION_VALUE_XYZ_ARRAY	&lvaSrc
	   )
{
	int iMinCount = lvaSrc.iCount;

	lvaDst.lvzOrigin = lvaSrc.lvzOrigin;
	for( int i = 0; i < iMinCount; i++ )
	{
		lvaDst.plvz[i] = lvaSrc.plvz[i];
	}
	lvaDst.iCount = iMinCount;
	return 1;
}

int
lvEnforceDistFromOrigin(
				 LOCATION_VALUE_XYZ &lpOrigins,
				 LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
				 int iMaxDist
				 )
{
	int newCount = lpaVariables.iCount;
	int iCurrValidCount = 0;

    int	MaxDistSqr = iMaxDist * iMaxDist;

	for( int i = 0; i < lpaVariables.iCount; i++ )
	{
		// Test to see if current location is greater than the 
		// maximum allowable distance from the original location
		// in the first image

		LOCATION_VALUE_XYZ &lvCurr = lpaVariables.plvz[i];

		int DistX = lvCurr.x - lpOrigins.x;
		int DistY = lvCurr.y - lpOrigins.y;
		int DistZ = lvCurr.z - lpOrigins.z;
		int DistSqr = (DistX * DistX) + (DistY * DistY) + (DistZ * DistZ);

		if( DistSqr <= MaxDistSqr )
		{
			// Point inside allowable radius, keep in array
			lpaVariables.plvz[iCurrValidCount] = lpaVariables.plvz[i];
			iCurrValidCount++;
		}
		else
		{
			newCount--;
		}
	}

	lpaVariables.iCount = newCount;
	
	return 1;
}


int
lvEnforceDistFromPoint(
				 LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
				 LOCATION_VALUE_XYZ &lvPoint,
				 int iDist
				 )
{
	int newCount = lpaVariables.iCount;
	int iCurrValidCount = 0;

    int	iMaxDistSqr = iDist * iDist;

	for( int i = 0; i < lpaVariables.iCount; i++ )
	{
		// Test to see if current location is greater than the 
		// maximum allowable distance from the original location
		// in the first image

		LOCATION_VALUE_XYZ &lvCurr = lpaVariables.plvz[i];

		int DistX = lvCurr.x - lvPoint.x;
		int DistY = lvCurr.y - lvPoint.y;
		int DistZ = lvCurr.z - lvPoint.z;
		int iDistSqr = (DistX * DistX) + (DistY * DistY) + (DistZ * DistZ);

		if( iDistSqr >= iMaxDistSqr )
		{
			// Point inside allowable radius, keep in array
			lpaVariables.plvz[iCurrValidCount] = lpaVariables.plvz[i];
			iCurrValidCount++;
		}
		else
		{
			newCount--;
		}
	}

	lpaVariables.iCount = newCount;
	
	return 1;
}

//
// lvEnforceDiffPoints
//
// Ensures that the points chosen by MoreTransformPoints are different
// from the fixed points chosen by SampleTransform
//
int
lvEnforceDiffPoints(
				 LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
				 LOCATION_VALUE_XYZ_COLLECTION &lpcFixedPoints,
				 int bgFlag									// 1 for basal ganglion to full brain, 0 for same size
															// matches
				 )
{
	int newCount = lpaVariables.iCount;
	int iCurrValidCount = 0;
	
	if ( bgFlag != 1 && bgFlag != 0 )
		return 0;

	for( int i = 0; i < lpaVariables.iCount; i++ )
	{
		// Test to see if the current point is different from
		// the fixed points
		LOCATION_VALUE_XYZ &lvCurr = lpaVariables.plvz[i];
		int sameFlag = 0;

		for( int j = 0; j < lpcFixedPoints.iCount; j++ )
		{
			LOCATION_VALUE_XYZ &lvFixed = lpcFixedPoints.plvza[j].plvz[0];

			if( bgFlag == 1)
			{
				if( (lvCurr.x == lvFixed.x - 50) &&
					(lvCurr.y == lvFixed.y - 60) &&
					(lvCurr.z == lvFixed.z - 50) )
				{
					sameFlag = 1;
					break;
				}
			}
			else
			{
				if( (lvCurr.x == lvFixed.x) &&
					(lvCurr.y == lvFixed.y) &&
					(lvCurr.z == lvFixed.z) )
				{
					sameFlag = 1;
					break;
				}
			}
		}

		if( sameFlag == 1)
		{
			newCount--;
		}
		else
		{
			// Points are different, keep in array
			lpaVariables.plvz[iCurrValidCount] = lpaVariables.plvz[i];
			iCurrValidCount++;
		}
	}
	
	lpaVariables.iCount = newCount;
	
	return 1;
}


int
lvEnforceNoDuplicateXY(
				 LOCATION_VALUE_XYZ_ARRAY &lpaVariables
				 )
{
	int i, j, iCurrValidCount;

	iCurrValidCount = 1;

	for( int i = 1; i < lpaVariables.iCount; i++ )
	{
		// Test to see if current variable is greater than
		// min dist away from all previous variables.

		LOCATION_VALUE_XYZ &lvCurr = lpaVariables.plvz[i];

		int bDuplicateXY = 0;

		for( int j = 0; j < iCurrValidCount && !bDuplicateXY; j++ )
		{
			LOCATION_VALUE_XYZ &lvPrev = lpaVariables.plvz[j];

			if( lvCurr.x == lvPrev.x && lvCurr.y == lvPrev.y )
			{
				bDuplicateXY = 1;
			}
		}

		if( !bDuplicateXY )
		{
			// Min dist is valid, keep
			lpaVariables.plvz[iCurrValidCount] = lpaVariables.plvz[i];
			iCurrValidCount++;
		}
	}

	lpaVariables.iCount = iCurrValidCount;

	return 1;
}


int
lvGetIndex(
		const LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
		const LOCATION_VALUE_XYZ &lvPoint
		)
{
	for( int i = 0; i < lpaVariables.iCount; i++ )
	{
		LOCATION_VALUE_XYZ &lvCurr = lpaVariables.plvz[i];
		if( lvPoint.x == lvCurr.x
			&& lvPoint.y == lvCurr.y
			&& lvPoint.z == lvCurr.z )
		{
			return i;
		}
	}
	return -1;
}

int
lvInitializeCollection(
					   const LOCATION_VALUE_XYZ_ARRAY	&lva,
					   LOCATION_VALUE_XYZ_COLLECTION	&lvc,
					   const int iCount
					   )
{
	lvc.iCount = lva.iCount;
	lvc.plvza = new LOCATION_VALUE_XYZ_ARRAY[lvc.iCount];
	assert( lvc.plvza );
	for( int i = 0; i < lva.iCount; i++ )
	{
		lvc.plvza[i].iCount = iCount;
		lvc.plvza[i].lvzOrigin = lva.lvzOrigin;
		lvc.plvza[i].plvz = new LOCATION_VALUE_XYZ[iCount];
		assert( lvc.plvza[i].plvz );
	}
	return 1;
}


int
lvInitializeCollection(
					   LOCATION_VALUE_XYZ_COLLECTION	&lvc,
					   const int iCollectionSize,
					   const int iArraySize
					   )
{
	lvc.iCount = iCollectionSize;
	lvc.plvza = new LOCATION_VALUE_XYZ_ARRAY[iCollectionSize];
	assert( lvc.plvza );
	for( int i = 0; i < iCollectionSize; i++ )
	{
		lvc.plvza[i].iCount = iArraySize;
		lvc.plvza[i].plvz = new LOCATION_VALUE_XYZ[iArraySize];
		assert( lvc.plvza[i].plvz );
	}
	return 1;
}


int
lvDeleteCollection(
				   LOCATION_VALUE_XYZ_COLLECTION	&lvc
				   )
{
	for( int i = 0; i < lvc.iCount; i++ )
	{
		if( lvc.plvza[i].plvz )
		{
			delete [] lvc.plvza[i].plvz;
		}
	}
	delete [] lvc.plvza;
	return 1;
}

int
lvShuffle(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables
			   )
{
	// Go through array 5 times
	for( int i = 0; i < 5; i++ )
	{
		// Shuffle each point to a random poitn
		for( int idx = 0; idx < lpaVariables.iCount; idx++ )
		{
			int idx2 = (rand()*lpaVariables.iCount)/(RAND_MAX + 1.0);
			while( idx2 == idx )
			{
				idx2 = (rand()*lpaVariables.iCount)/(RAND_MAX + 1.0);
			}
			LOCATION_VALUE_XYZ lvTmp = lpaVariables.plvz[idx];
			lpaVariables.plvz[idx] = lpaVariables.plvz[idx2];
			lpaVariables.plvz[idx2] = lvTmp;
		}
	}
	return 1;
}
