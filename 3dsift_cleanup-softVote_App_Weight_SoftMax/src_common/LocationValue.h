
#ifndef __LOCATIONVALUE_H__
#define __LOCATIONVALUE_H__

//
// Location plus a probability, used in sampling
//
typedef struct _LOCATION_VALUE
{
	int		iRow;
	int		iCol;
	float	fProb;
} LOCATION_VALUE;

//
// Array of LOCATION_VALUE pairs plus entropy
//
typedef struct _LOCATION_VALUE_ARRAY
{
	LOCATION_VALUE	*plvs;
	int				ilpCount;
	int				iRow;	// Original row value
	int				iCol;	// Original col value
	float			fEntropy;
} LOCATION_VALUE_ARRAY;

//
// Collection of LOCATION_VALUE_ARRAY
//
typedef struct _LOCATION_VALUE_COLLECTION
{
	LOCATION_VALUE_ARRAY *plpas;
	int					ilpaCount;
} LOCATION_VALUE_COLLECTION;


//
// (xyz location,value), array, collection of arrays
//

typedef struct _LOCATION_VALUE_XYZ
{
	int		x;
	int		y;
	int		z;
	float	fValue;
} LOCATION_VALUE_XYZ;

typedef struct _LOCATION_VALUE_XYZ_ARRAY
{
	LOCATION_VALUE_XYZ	*plvz;		// Array
	int					iCount;	// Count
	LOCATION_VALUE_XYZ	lvzOrigin;	// Original
} LOCATION_VALUE_XYZ_ARRAY;

typedef struct _LOCATION_VALUE_XYZ_COLLECTION
{
	LOCATION_VALUE_XYZ_ARRAY *plvza;
	int					iCount;
} LOCATION_VALUE_XYZ_COLLECTION;

int
lvInitNeighboursFullyConnected(
						const LOCATION_VALUE_XYZ_ARRAY &lvaVariables,
						const int iVariables,
						unsigned char *pucNeighbours
						);

int
lvEnforceMinDist(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
			   int iMinDist
			   );

int
lvEnforceMinAbsValue(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
			   float fAbsValue
			   );

int
lvWriteLocationValueXYZArray(
						   LOCATION_VALUE_XYZ_ARRAY &LPA,
						   char *pcFileName
						   );

int
lvReadLocationValueXYZArray(
						   LOCATION_VALUE_XYZ_ARRAY &lva,
						   char *pcFileName
					   );

int
lvCopy(
	   LOCATION_VALUE_XYZ_ARRAY			&lvaDst,
	   const LOCATION_VALUE_XYZ_ARRAY	&lvaSrc
	   );

int
lvSortHighLow(
			  LOCATION_VALUE_XYZ_ARRAY &lva
			  );

int
lvSortLowHigh(
			  LOCATION_VALUE_XYZ_ARRAY &lva
			  );

int
lvEnforceRangeX(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
			   int iMin,
			   int iMax
			   );

int
lvEnforceRangeY(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
			   int iMin,
			   int iMax
			   );

int
lvEnforceRangeZ(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
			   int iMin,
			   int iMax
			   );

//
// lvShuffle()
//
// Shuffle order of points
//
int
lvShuffle(
			   LOCATION_VALUE_XYZ_ARRAY &lpaVariables
			   );

//
// lvEnforceDistFromOrigin()
//
// Enforces the constraint that all candidates in lpaVariables be
// within distance iMaxDist from the orgin of lpaVariables.
//
// Outputs:
//	1) possibly removes candidates from lpaVariables
//	2) updates the count field of lpaVariable
//
int
lvEnforceDistFromOrigin(
				 LOCATION_VALUE_XYZ &lpOrigins,
				 LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
				 int iMaxDist
				 );

//
// lvEnforceDistFromPoint()
//
// Enforces the constraint that all variables in lpaVariables
// be a distance iMaxDist from lvPoint.
//
int
lvEnforceDistFromPoint(
				 LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
				 LOCATION_VALUE_XYZ &lvPoint,
				 int iDist
				 );

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
				 );

//
// lvEnforceNoDuplicateXY()
// 
// Ensures no duplicate XY coordinates exist. Removes
// those lowest on the list.
//
int
lvEnforceNoDuplicateXY(
				 LOCATION_VALUE_XYZ_ARRAY &lpaVariables
				 );

//
// lvGetIndex()
//
// Returns the index of point lvPoint in the array lpaVariables. Returns -1
// if lvPoint could not be found.
//
int
lvGetIndex(
		const LOCATION_VALUE_XYZ_ARRAY &lpaVariables,
		const LOCATION_VALUE_XYZ &lvPoint
		);

//
// lvDistSqr()
//
// Returns the squared Euclidean distance between lv1.(x,y,z) and lv2.(x,y,z)
//
int
lvDistSqr(
		  const LOCATION_VALUE_XYZ &lv1,
		  const LOCATION_VALUE_XYZ &lv2
		  );
float
lvDistSqrFloat(
		  const LOCATION_VALUE_XYZ &lv1,
		  const LOCATION_VALUE_XYZ &lv2
		  );

//
// lvInitializeCollection()
//
// Initializes a LOCATION_VALUE_XYZ_COLLECTION such that each
// LOCATION_VALUE_XYZ element of lva becomes an LOCATION_VALUE_XYZ_ARRAY
// in lvc with iCount values.
//
int
lvInitializeCollection(
					   const LOCATION_VALUE_XYZ_ARRAY	&lva,
					   LOCATION_VALUE_XYZ_COLLECTION	&lvc,
					   const int						iCount = 10
					   );

//
// lvInitializeCollection()
//
// Initializes lvc with iCollectionSize arrays, each of size iArraySize.
//
int
lvInitializeCollection(
					   LOCATION_VALUE_XYZ_COLLECTION	&lvc,
					   const int iCollectionSize,
					   const int iArraySize
					   );

//
// lvDeleteCollection()
//
// Deallocate all memory associated with lvC
//
int
lvDeleteCollection(
				   LOCATION_VALUE_XYZ_COLLECTION	&lvc
				   );

#endif
