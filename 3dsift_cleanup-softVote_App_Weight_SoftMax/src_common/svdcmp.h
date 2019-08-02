
#ifndef __SVDCMP_H__
#define __SVDCMP_H__

void
svdcmp(
	   double* a[],
	   int m,
	   int n,
	   double w[],
	   double* v[]
	   );

void
svdcmp_iterations(
	   double* a[],
	   int m,
	   int n,
	   double w[],
	   double* v[],
	   int iMaxIts = 30
	   );

void
reorder_descending(
	   int m,
	   int n,
	   double w[],
	   double* v[]
	   );

int
mossSVD(double *U, double *W, double *V, double *matx, int M, int N);

int
mossPseudoInverse(double *inv, double *matx, int M, int N);

#endif
