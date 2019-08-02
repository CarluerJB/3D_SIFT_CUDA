/*
* $Log: svdcmp.cc,v $
* Revision 1.5  2000/06/13 20:04:01  jbrandt
* updated to use iostream
*
* Revision 1.4  2000/03/15 15:34:31  jbrandt
* Changed Math.h to MathExtras.h
*
* Revision 1.3  1994/01/12 14:53:25  jbrandt
* Deleted libc.h, added stdlib.h and changed math.h to Math.h
*
* Revision 1.2  1993/02/05  16:54:39  jbrandt
* *** empty log message ***
*
*/

//#include "MathExtras.h"
#include <math.h>
//#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "svdcmp.h"

/*
PYTHAG computes sqrt(a^{2} + b^{2}) without destructive overflow or underflow.
*/
static double at, bt, ct;
#define PYTHAG(a, b) ((at = fabs(a)) > (bt = fabs(b)) ? \
(ct = bt/at, at*sqrt(1.0+ct*ct)): (bt ? (ct = at/bt, bt*sqrt(1.0+ct*ct)): 0.0))

static double maxarg1, maxarg2;
#define MAX(a, b) (maxarg1 = (a), maxarg2 = (b), (maxarg1) > (maxarg2) ? \
(maxarg1) : (maxarg2))

#define SIGN(a, b) ((b) < 0.0 ? -fabs(a): fabs(a))

void error(char error_text[])
/* Numerical Recipes standard error handler.                        */
{
	
    fprintf(stderr,"Numerical Recipes run-time error...\n");
    fprintf(stderr, "%s %s\n", error_text, "...now exiting to system...");
    exit(1);
}

/*
*Note: this is garaunteed to work for m==n
*/
void
reorder_descending(
	   int m,
	   int n,
	   double w[],
	   double* v[]
	   )
{
	// Order N*N - use quicksort if too slow

	for( int i = 0; i < n; i++ )
	{
		double	dLargestValue = w[i];
		int		iLargestIndex = i;
		for( int j = i + 1; j < n; j++ )
		{
			if( fabs( w[j] ) > fabs( dLargestValue ) )
			{
				dLargestValue = w[j];
				iLargestIndex = j;
			}
		}

		// Swap eigen value and vector at i with value at largest index

		double dTemp;

		dTemp = w[i];
		w[i] = w[iLargestIndex];
		w[iLargestIndex] = dTemp;

		// Swap eigen vector columns

		for( int r = 0; r < n; r++ )
		{
			dTemp = v[r][i];
			v[r][i] = v[r][iLargestIndex];
			v[r][iLargestIndex] = dTemp;
		}
	}
}

/*
Given a matrix a[m][n], this routine computes its singular value
decomposition, A = U*W*V^{T}.  The matrix U replaces a on output.
The diagonal matrix of singular values W is output as a vector w[n].
The matrix V (not the transpose V^{T}) is output as v[n][n].
m must be greater or equal to n;  if it is smaller, then a should be
filled up to square with zero rows.
*/
void
svdcmp(
	   double* a[],
	   int m,
	   int n,
	   double w[],
	   double* v[]
	   )
{
    int flag, i, its, j, jj, k, l, nm;
    double c, f, h, s, x, y, z;
    double anorm = 0.0, g = 0.0, scale = 0.0;
	
    if (m < n)
		error("SVDCMP: Matrix A must be augmented with extra rows of zeros.");
    double* rv1 = new double [n];
	
    /* Householder reduction to bidiagonal form.                        */
    for (i = 0; i < n; i++)
	{
        l = i + 1;
        rv1[i] = scale*g;
        g = s = scale = 0.0;
        if (i < m)
		{
            for (k = i; k < m; k++)
			{
				scale += fabs(a[k][i]);
			}
            if (scale)
			{
                for (k = i; k < m; k++)
				{
                    a[k][i] /= scale;
                    s += a[k][i]*a[k][i];
				}
                f = a[i][i];
                g = -SIGN(sqrt(s), f);
                h = f*g - s;
                a[i][i] = f - g;
                if (i != n - 1)
				{
                    for (j = l; j < n; j++)
					{
                        for (s  = 0.0, k = i; k < m; k++)
							s += a[k][i]*a[k][j];
                        f = s/h;
                        for ( k = i; k < m; k++)
							a[k][j] += f*a[k][i];
					}
				}
                for (k = i; k < m; k++)
					a[k][i] *= scale;
			}
		}
        w[i] = scale*g;
        g = s= scale = 0.0;
        if (i < m && i != n - 1)
		{
            for (k = l; k < n; k++)
				scale += fabs(a[i][k]);
            if (scale)
			{
                for (k = l; k < n; k++)
				{
                    a[i][k] /= scale;
                    s += a[i][k]*a[i][k];
				}
                f = a[i][l];
                g = -SIGN(sqrt(s), f);
                h = f*g - s;
                a[i][l] = f - g;
                for (k = l; k < n; k++)
					rv1[k] = a[i][k]/h;
                if (i != m - 1)
				{
                    for (j = l; j < m; j++)
					{
                        for (s = 0.0, k = l; k < n; k++)
							s += a[j][k]*a[i][k];
                        for (k = l; k < n; k++)
							a[j][k] += s*rv1[k];
					}
				}
                for (k = l; k < n; k++)
					a[i][k] *= scale;
			}
		}
        anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}
    /* Accumulation of right-hand transformations.                      */
    for (i = n - 1; 0 <= i; i--)
	{
        if (i < n - 1)
		{
            if (g)
			{
                for (j = l; j < n; j++)
					v[j][i] = (a[i][j]/a[i][l])/g;
				/* Double division to avoid possible underflow:       */
                for (j = l; j < n; j++)
				{
                    for (s = 0.0, k = l; k < n; k++)
						s += a[i][k]*v[k][j];
                    for (k = l; k < n; k++)
						v[k][j] += s*v[k][i];
				}
			}
            for (j = l; j < n; j++)
				v[i][j] = v[j][i] = 0.0;
		}
        v[i][i] = 1.0;
        g = rv1[i];
        l = i;
	}
    /* Accumulation of left-hand transformations.                       */
    for (i = n - 1; 0 <= i; i--)
	{
        l = i + 1;
        g = w[i];
        if (i < n - 1)
			for (j = l; j < n; j++)
				a[i][j] = 0.0;
			if (g)
			{
				g = 1.0/g;
				if (i != n - 1)
				{
					for (j = l; j < n; j++)
					{
						for (s = 0.0, k = l; k < m; k++)
							s += a[k][i]*a[k][j];
						f = (s/a[i][i])*g;
						for (k = i; k < m; k++)
							a[k][j] += f*a[k][i];
					}
				}
				for (j = i; j < m; j++)
					a[j][i] *= g;
			}
			else
				for (j = i; j < m; j++)
					a[j][i] = 0.0;
				++a[i][i];
	}
    /* Diagonalization of the bidiagonal form.                          */
    for (k = n - 1; 0 <= k; k--)        /* Loop over singular values.   */
	{
        for (its = 0; its < 30; its++)  /* Loop over allowed iterations.*/
		{
            flag = 1;
            for (l = k; 0 <= l; l--)    /* Test for splitting:          */
			{
                nm = l - 1;             /* Note that rv1[0] is always zero.*/
                if (fabs(rv1[l]) + anorm == anorm)
				{
                    flag = 0;
                    break;
				}
                if (fabs(w[nm]) + anorm == anorm)
					break;
			}
            if (flag)
			{
                c = 0.0;                /* Cancellation of rv1[l], if l>0:*/
                s = 1.0;
                for (i = l; i <= k; i++) {
                    f = s*rv1[i];
                    if (fabs(f) + anorm != anorm)
					{
                        g = w[i];
                        h = PYTHAG(f, g);
                        w[i] = h;
                        h = 1.0/h;
                        c = g*h;
                        s = (-f*h);
                        for (j = 0; j < m; j++)
						{
                            y = a[j][nm];
                            z = a[j][i];
                            a[j][nm] = y*c + z*s;
                            a[j][i]  = z*c - y*s;
						}
					}
				}
			}
            z = w[k];
            if (l == k)         /* Convergence.                         */
			{
                if (z < 0.0)    /* Singular value is made non-negative. */
				{
                    w[k] = -z;
                    for (j = 0; j < n; j++)
						v[j][k] = (-v[j][k]);
				}
                break;
			}
            if (its == 29)
				error("No convergence in 30 SVDCMP iterations.");
            x = w[l];           /* Shift from bottom 2-by-2 minor.      */
            nm = k - 1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z)*(y + z) + (g - h)*(g + h))/(2.0*h*y);
            g = PYTHAG(f, 1.0);
            f = ((x - z)*(x + z) + h*((y/(f + SIGN(g, f))) - h))/x;
            /* Next QR transformation:                                  */
            c = s = 1.0;
            for (j = l; j <= nm; j++)
			{
                i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s*g;
                g = c*g;
                z = PYTHAG(f, h);
                rv1[j] = z;
                c = f/z;
                s = h/z;
                f = x*c + g*s;
                g = g*c - x*s;
                h = y*s;
                y = y*c;
                for (jj = 0; jj < n;  jj++)
				{
                    x = v[jj][j];
                    z = v[jj][i];
                    v[jj][j] = x*c + z*s;
                    v[jj][i] = z*c - x*s;
				}
                z = PYTHAG(f, h);
                w[j] = z;       /* Rotation can be arbitrary if z = 0.  */
                if (z)
				{
                    z = 1.0/z;
                    c = f*z;
                    s = h*z;
				}
                f = (c*g) + (s*y);
                x = (c*y) - (s*g);
                for (jj = 0; jj < m; jj++)
				{
                    y = a[jj][j];
                    z = a[jj][i];
                    a[jj][j] = y*c + z*s;
                    a[jj][i] = z*c - y*s;
				}
			}
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
          }
      }
	  delete rv1;
}

void
svdcmp_iterations(
	   double* a[],
	   int m,
	   int n,
	   double w[],
	   double* v[],
	   int iMaxIts
	   )
{
    int flag, i, its, j, jj, k, l, nm;
    double c, f, h, s, x, y, z;
    double anorm = 0.0, g = 0.0, scale = 0.0;
	
    if (m < n)
	{
		error("SVDCMP: Matrix A must be augmented with extra rows of zeros.");
	}
    double* rv1 = new double [n];
	
    /* Householder reduction to bidiagonal form.                        */
    for (i = 0; i < n; i++)
	{
        l = i + 1;
        rv1[i] = scale*g;
        g = s = scale = 0.0;
        if (i < m)
		{
            for (k = i; k < m; k++)
			{
				scale += fabs(a[k][i]);
			}
            if (scale)
			{
                for (k = i; k < m; k++)
				{
                    a[k][i] /= scale;
                    s += a[k][i]*a[k][i];
				}
                f = a[i][i];
                g = -SIGN(sqrt(s), f);
                h = f*g - s;
                a[i][i] = f - g;
                if (i != n - 1)
				{
                    for (j = l; j < n; j++)
					{
                        for (s  = 0.0, k = i; k < m; k++)
							s += a[k][i]*a[k][j];
                        f = s/h;
                        for ( k = i; k < m; k++)
							a[k][j] += f*a[k][i];
					}
				}
                for (k = i; k < m; k++)
					a[k][i] *= scale;
			}
		}
        w[i] = scale*g;
        g = s= scale = 0.0;
        if (i < m && i != n - 1)
		{
            for (k = l; k < n; k++)
				scale += fabs(a[i][k]);
            if (scale)
			{
                for (k = l; k < n; k++)
				{
                    a[i][k] /= scale;
                    s += a[i][k]*a[i][k];
				}
                f = a[i][l];
                g = -SIGN(sqrt(s), f);
                h = f*g - s;
                a[i][l] = f - g;
                for (k = l; k < n; k++)
					rv1[k] = a[i][k]/h;
                if (i != m - 1)
				{
                    for (j = l; j < m; j++)
					{
                        for (s = 0.0, k = l; k < n; k++)
							s += a[j][k]*a[i][k];
                        for (k = l; k < n; k++)
							a[j][k] += s*rv1[k];
					}
				}
                for (k = l; k < n; k++)
					a[i][k] *= scale;
			}
		}
        anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}
    /* Accumulation of right-hand transformations.                      */
    for (i = n - 1; 0 <= i; i--)
	{
        if (i < n - 1)
		{
            if (g)
			{
                for (j = l; j < n; j++)
					v[j][i] = (a[i][j]/a[i][l])/g;
				/* Double division to avoid possible underflow:       */
                for (j = l; j < n; j++)
				{
                    for (s = 0.0, k = l; k < n; k++)
						s += a[i][k]*v[k][j];
                    for (k = l; k < n; k++)
						v[k][j] += s*v[k][i];
				}
			}
            for (j = l; j < n; j++)
				v[i][j] = v[j][i] = 0.0;
		}
        v[i][i] = 1.0;
        g = rv1[i];
        l = i;
	}
    /* Accumulation of left-hand transformations.                       */
    for (i = n - 1; 0 <= i; i--)
	{
        l = i + 1;
        g = w[i];
        if (i < n - 1)
			for (j = l; j < n; j++)
				a[i][j] = 0.0;
			if (g)
			{
				g = 1.0/g;
				if (i != n - 1)
				{
					for (j = l; j < n; j++)
					{
						for (s = 0.0, k = l; k < m; k++)
							s += a[k][i]*a[k][j];
						f = (s/a[i][i])*g;
						for (k = i; k < m; k++)
							a[k][j] += f*a[k][i];
					}
				}
				for (j = i; j < m; j++)
					a[j][i] *= g;
			}
			else
				for (j = i; j < m; j++)
					a[j][i] = 0.0;
				++a[i][i];
	}
    /* Diagonalization of the bidiagonal form.                          */
    for (k = n - 1; 0 <= k; k--)        /* Loop over singular values.   */
	{
        for (its = 0; its < iMaxIts; its++)  /* Loop over allowed iterations.*/
		{
            flag = 1;
            for (l = k; 0 <= l; l--)    /* Test for splitting:          */
			{
                nm = l - 1;             /* Note that rv1[0] is always zero.*/
                if (fabs(rv1[l]) + anorm == anorm)
				{
                    flag = 0;
                    break;
				}
                if (fabs(w[nm]) + anorm == anorm)
					break;
			}
            if (flag)
			{
                c = 0.0;                /* Cancellation of rv1[l], if l>0:*/
                s = 1.0;
                for (i = l; i <= k; i++) {
                    f = s*rv1[i];
                    if (fabs(f) + anorm != anorm)
					{
                        g = w[i];
                        h = PYTHAG(f, g);
                        w[i] = h;
                        h = 1.0/h;
                        c = g*h;
                        s = (-f*h);
                        for (j = 0; j < m; j++)
						{
                            y = a[j][nm];
                            z = a[j][i];
                            a[j][nm] = y*c + z*s;
                            a[j][i]  = z*c - y*s;
						}
					}
				}
			}
            z = w[k];
            if (l == k)         /* Convergence.                         */
			{
                if (z < 0.0)    /* Singular value is made non-negative. */
				{
                    w[k] = -z;
                    for (j = 0; j < n; j++)
						v[j][k] = (-v[j][k]);
				}
                break;
			}
            if (its == iMaxIts-1)
				error("No convergence in 30 SVDCMP iterations.");
            x = w[l];           /* Shift from bottom 2-by-2 minor.      */
            nm = k - 1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z)*(y + z) + (g - h)*(g + h))/(2.0*h*y);
            g = PYTHAG(f, 1.0);
            f = ((x - z)*(x + z) + h*((y/(f + SIGN(g, f))) - h))/x;
            /* Next QR transformation:                                  */
            c = s = 1.0;
            for (j = l; j <= nm; j++)
			{
                i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s*g;
                g = c*g;
                z = PYTHAG(f, h);
                rv1[j] = z;
                c = f/z;
                s = h/z;
                f = x*c + g*s;
                g = g*c - x*s;
                h = y*s;
                y = y*c;
                for (jj = 0; jj < n;  jj++)
				{
                    x = v[jj][j];
                    z = v[jj][i];
                    v[jj][j] = x*c + z*s;
                    v[jj][i] = z*c - x*s;
				}
                z = PYTHAG(f, h);
                w[j] = z;       /* Rotation can be arbitrary if z = 0.  */
                if (z)
				{
                    z = 1.0/z;
                    c = f*z;
                    s = h*z;
				}
                f = (c*g) + (s*y);
                x = (c*y) - (s*g);
                for (jj = 0; jj < m; jj++)
				{
                    y = a[jj][j];
                    z = a[jj][i];
                    a[jj][j] = y*c + z*s;
                    a[jj][i] = z*c - y*s;
				}
			}
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
          }
      }
	  delete rv1;
}


#define IMIN(m,n) (m < n ? m : n)
#define FMAX(m,n) (m > n ? m : n)

int 
_mossNRsvdcmp(double **a, int m, int n, double w[], double **v) {
  char me[]="_mossNRsvdcmp", err[128];
  //double pythag(double a, double b);
  int flag,i,its,j,jj,k,l,nm;
  double anorm,c,f,g,h,s,scale,x,y,z,*rv1;
  
  //rv1=vector(1,n);
  rv1=new double[n+1];
  g=scale=anorm=0.0;
  for (i=1;i<=n;i++) {
    l=i+1;
    rv1[i]=scale*g;
    g=s=scale=0.0;
    if (i <= m) {
      for (k=i;k<=m;k++) scale += fabs(a[k][i]);
      if (scale) {
	for (k=i;k<=m;k++) {
	  a[k][i] /= scale;
	  s += a[k][i]*a[k][i];
	}
	f=a[i][i];
	g = -SIGN(sqrt(s),f);
	h=f*g-s;
	a[i][i]=f-g;
	for (j=l;j<=n;j++) {
	  for (s=0.0,k=i;k<=m;k++) s += a[k][i]*a[k][j];
	  f=s/h;
	  for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
	}
	for (k=i;k<=m;k++) a[k][i] *= scale;
      }
    }
    w[i]=scale *g;
    g=s=scale=0.0;
    if (i <= m && i != n) {
      for (k=l;k<=n;k++) scale += fabs(a[i][k]);
      if (scale) {
	for (k=l;k<=n;k++) {
	  a[i][k] /= scale;
	  s += a[i][k]*a[i][k];
	}
	f=a[i][l];
	g = -SIGN(sqrt(s),f);
	h=f*g-s;
	a[i][l]=f-g;
	for (k=l;k<=n;k++) rv1[k]=a[i][k]/h;
	for (j=l;j<=m;j++) {
	  for (s=0.0,k=l;k<=n;k++) s += a[j][k]*a[i][k];
	  for (k=l;k<=n;k++) a[j][k] += s*rv1[k];
	}
	for (k=l;k<=n;k++) a[i][k] *= scale;
      }
    }
    anorm=FMAX(anorm,(fabs(w[i])+fabs(rv1[i])));
  }
  for (i=n;i>=1;i--) {
    if (i < n) {
      if (g) {
	for (j=l;j<=n;j++)
	  v[j][i]=(a[i][j]/a[i][l])/g;
	for (j=l;j<=n;j++) {
	  for (s=0.0,k=l;k<=n;k++) s += a[i][k]*v[k][j];
	  for (k=l;k<=n;k++) v[k][j] += s*v[k][i];
	}
      }
      for (j=l;j<=n;j++) v[i][j]=v[j][i]=0.0;
    }
    v[i][i]=1.0;
    g=rv1[i];
    l=i;
  }
  for (i=IMIN(m,n);i>=1;i--) {
    l=i+1;
    g=w[i];
    for (j=l;j<=n;j++) a[i][j]=0.0;
    if (g) {
      g=1.0/g;
      for (j=l;j<=n;j++) {
	for (s=0.0,k=l;k<=m;k++) s += a[k][i]*a[k][j];
	f=(s/a[i][i])*g;
	for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
      }
      for (j=i;j<=m;j++) a[j][i] *= g;
    } else for (j=i;j<=m;j++) a[j][i]=0.0;
    ++a[i][i];
  }
  for (k=n;k>=1;k--) {
    for (its=1;its<=30;its++) {
      flag=1;
      for (l=k;l>=1;l--) {
	nm=l-1;
	if ((double)(fabs(rv1[l])+anorm) == anorm) {
	  flag=0;
	  break;
	}
	if ((double)(fabs(w[nm])+anorm) == anorm) break;
      }
      if (flag) {
	c=0.0;
	s=1.0;
	for (i=l;i<=k;i++) {
	  f=s*rv1[i];
	  rv1[i]=c*rv1[i];
	  if ((double)(fabs(f)+anorm) == anorm) break;
	  g=w[i];
	  h=PYTHAG(f,g);
	  w[i]=h;
	  h=1.0/h;
	  c=g*h;
	  s = -f*h;
	  for (j=1;j<=m;j++) {
	    y=a[j][nm];
	    z=a[j][i];
	    a[j][nm]=y*c+z*s;
	    a[j][i]=z*c-y*s;
	  }
	}
      }
      z=w[k];
      if (l == k) {
	if (z < 0.0) {
	  w[k] = -z;
	  for (j=1;j<=n;j++) v[j][k] = -v[j][k];
	}
	break;
      }
      if (its == 30) {
	sprintf(err, "%s: no convergence in 30 svdcmp iterations", me);
	return 1;
      }
      x=w[l];
      nm=k-1;
      y=w[nm];
      g=rv1[nm];
      h=rv1[k];
      f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
      g=PYTHAG(f,1.0);
      f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
      c=s=1.0;
      for (j=l;j<=nm;j++) {
	i=j+1;
	g=rv1[i];
	y=w[i];
	h=s*g;
	g=c*g;
	z=PYTHAG(f,h);
	rv1[j]=z;
	c=f/z;
	s=h/z;
	f=x*c+g*s;
	g = g*c-x*s;
	h=y*s;
	y *= c;
	for (jj=1;jj<=n;jj++) {
	  x=v[jj][j];
	  z=v[jj][i];
	  v[jj][j]=x*c+z*s;
	  v[jj][i]=z*c-x*s;
	}
	z=PYTHAG(f,h);
	w[j]=z;
	if (z) {
	  z=1.0/z;
	  c=f*z;
	  s=h*z;
	}
	f=c*g+s*y;
	x=c*y-s*g;
	for (jj=1;jj<=m;jj++) {
	  y=a[jj][j];
	  z=a[jj][i];
	  a[jj][j]=y*c+z*s;
	  a[jj][i]=z*c-y*s;
	}
      }
      rv1[l]=0.0;
      rv1[k]=f;
      w[k]=x;
    }
  }
  delete [] rv1;
  //free_vector(rv1,1,n);
  return 0;
}
#undef NRANSI

int
mossSVD(double *U, double *W, double *V, double *matx, int M, int N) {
  char me[]="mossSVD", err[128];
  double **nrU, *nrW, **nrV;
  int problem, i;

  /* <rant> The stupid tricks we play in order to interface with
     incredibly annoying Numerical Recipes code.  When I was a Cornell
     undergrad, I was interested in a programming job in the Astronomy
     department.  The first thing that Saul Teukolsky (one of the
     Numerical Recipes authors) asked was "What's your GPA?".  I was
     confused.  If programming ability is the real question at hand,
     this was about as sensible as asking "How much do you weigh?"  As
     soon as I admitted to getting a B+ in a previous previous physics
     class, he waved me away.  His loss.  This alone is sufficient
     evidence that he is a narrow-minded pissant in matters of
     progamming style.  This tired claptrap (which the Numerical
     Recipes authors spill) about how arrays make more sense when they
     are one-offset instead of zero-offset is an infuriatingly thin
     veil for the fact that these people are dinosaurs and FORTRAN
     apologists at heart.  For them to produce human-readable bug-free
     idiomatic C code would be as easy as performing self-trepanation
     after drinking two pots of coffee. </rant> */

  /* allocate arrays for the Numerical Recipes code to write into */
  nrU = (double **)malloc((M+1)*sizeof(double*));
  nrW = (double *)malloc((N+1)*sizeof(double));
  nrV = (double **)malloc((N+1)*sizeof(double*));
  problem = !(nrU && nrW && nrV);
  if (!problem) {
    problem = 0;
    for (i=1; i<=M; i++) {
      nrU[i] = (double *)malloc((N+1)*sizeof(double));
      problem |= !nrU[i];
    }
    for(i=1; i<=N; i++) {
      nrV[i] = (double *)malloc((N+1)*sizeof(double));
      problem |= !nrV[i];
    }
  }
  if (problem) {
    sprintf(err, "%s: couldn't allocate arrays", me);
	return 1;
  }

  /* copy from given matx into nrU */
  for (i=1; i<=M; i++) {
    memcpy(&(nrU[i][1]), matx + N*(i-1), N*sizeof(double));
  }
  
  /*
  printf("%s: given matx:\n", me);
  for (i=1; i<=M; i++) {
    printf("%s:", me);
    for (j=1; j<=N; j++) {
      printf(" %g", nrU[i][j]);
    }
    printf("\n");
  }
  printf("%s:\n", me);
  */

  /* HERE IT IS: do SVD */
  if (_mossNRsvdcmp(nrU, M, N, nrW, nrV)) {
    sprintf(err, "%s: trouble in core SVD calculation", me);
    return 1;
  }
  /*
  printf("%s: svdcmp returned U:\n", me);
  for (i=1; i<=M; i++) {
    for (j=1; j<=N; j++) {
      printf(" %g", -nrU[i][j]);
    }
    printf("\n");
  }
  printf("%s:\n", me);
  printf("%s: svdcmp returned W:\n", me);
  for (i=1; i<=N; i++) {
    printf(" %g", nrW[i]);
  }
  printf("\n");
  printf("%s:\n", me);
  printf("%s: svdcmp returned V:\n", me);
  for (i=1; i<=N; i++) {
    for (j=1; j<=N; j++) {
      printf(" %g", -nrV[i][j]);
    }
    printf("\n");
  }
  printf("%s:\n", me);
  */
  
  /* copy results into caller's arrays */
  for (i=1; i<=M; i++) {
    memcpy(U + N*(i-1), &(nrU[i][1]), N*sizeof(double));
  }
  memcpy(W, &(nrW[1]), N*sizeof(double));
  for (i=1; i<=N; i++) {
    memcpy(V + N*(i-1), &(nrV[i][1]), N*sizeof(double));
  }

  /*
  printf("%s: we will return U:\n", me);
  for (i=0; i<=M-1; i++) {
    for (j=0; j<=N-1; j++) {
      printf(" %g", U[j+N*i]);
    }
    printf("\n");
  }
  printf("%s:\n", me);
  printf("%s: we will return W:\n", me);
  for (i=0; i<=N-1; i++) {
    printf(" %g", W[i]);
  }
  printf("\n");
  printf("%s:\n", me);
  printf("%s: we will return V:\n", me);
  for (i=0; i<=N-1; i++) {
    for (j=0; j<=N-1; j++) {
      printf(" %g", V[j+N*i]);
    }
    printf("\n");
  }
  printf("%s:\n", me);
  */
  
  /* free Numerical Recipes arrays */
  for (i=1; i<=M; i++)
    free(nrU[i]);
  free(nrU);
  free(nrW);
  for (i=1; i<=N; i++)
    free(nrV[i]);
  free(nrV);
  
  return 0;
}

#define MOSS_TINYVAL 0.0000000001

int
mossPseudoInverse(double *inv, double *matx, int M, int N) {
  char me[]="_mossPseudoInverse", err[128];
  double *U, *W, *V, ans;
  int problem, i, j, k;

  /*
  printf("%s: given M=%d, N=%d, matx=:\n", me, M, N);
  for (i=0; i<=M-1; i++) {
    printf("%s:", me);
    for (j=0; j<=N-1; j++) {
      printf(" %g", matx[j + N*i]);
    }
    printf("\n");
  }
  printf("%s:\n", me);
  */
  U = (double *)malloc(M*N*sizeof(double));
  W = (double *)malloc(N*sizeof(double));
  V = (double *)malloc(N*N*sizeof(double));
  if (!(U && W && V)) {
    sprintf(err, "%s: couldn't alloc matrices", me);
    return 1;
  }
  if (mossSVD(U, W, V, matx, M, N)) {
    sprintf(err, "%s: trouble in SVD computation", me);
    return 1;
  }
  problem = 0;
  for (i=0; i<=N-1; i++) {
    if (fabs(W[i]) < MOSS_TINYVAL) {
      sprintf(err, "%s: abs(W[%d]) = %g < %g = tiny", 
	      me, i, fabs(W[i]), MOSS_TINYVAL);
          problem = 1;
    }
  }
  if (problem) {
    sprintf(err, "%s: no pseudo-inverse due to small singular values", me);
    return 1;
  }
  for (i=0; i<=N-1; i++) {
    for (j=0; j<=M-1; j++) {
      ans = 0;
      for (k=0; k<=N-1; k++) {
	/* in V: row fixed at i, k goes through columns */
	/* in U^T: column fixed at j, k goes through rows ==>
	   in U: row fixed at j, k goes through columns */
	ans += V[k + N*i]*U[k + N*j]/W[k];
      }
      inv[j + M*i] = ans;
    }
  }
  free(U);
  free(W);
  free(V);
  /*
  printf("%s: returning inv:\n", me);
  for (i=0; i<=N-1; i++) {
    for (j=0; j<=M-1; j++) {
      printf(" %g", inv[j + M*i]);
    }
    printf("\n");
  }
  printf("%s:\n", me);
  */
  return 0;
}

