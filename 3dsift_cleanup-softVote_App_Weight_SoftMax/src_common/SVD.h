
#include <math.h>

#define _svd_max(a,b) (a > b ? a : b)

#define _svd_min(a,b) (a > b ? b : a)

#define pythag(a,b) (sqrt(a*a + b*b))

/*
Sorts d and v on decreasing values of d.
* \param d  returns the eigenvalues of a.
* \param v  is a matrix whose columns contain, the normalized eigenvectors
*/
template <class T, int N>
inline void SortEigenDecomp(T w[N], T v[N][N])
{
	T t;
	for(int i=0; i<N; i++)
		for(int j=i+1; j<N; j++)
			if(w[i] < w[j])
			{
				//swap eigenvalues
				t=w[j];    w[j]=w[i]; w[i]=t;
				//swap eigenvectors
				for(int k=0; k<N; k++)
				{
					t=v[k][j];    v[k][j]=v[k][i];    v[k][i]=t;
				}
			}
}


/*
Given a matrix mat[1..m][1..n], this routine computes its singular value 
decomposition,
A = U.Q.Vt. The matrix U replaces A on output. The diagonal matrix of 
singular values W is output
as a std::vector w[1..n]. The matrix V (not the transpose Vt) is output 
as v[1..n][1..n]).

//(C) Copr. 1986-92 Numerical Recipes Software -0).
*/
template <class T, int M, int N>
inline void SingularValueDecomp(
								T mat[M][N],
								T w[N],
								T v[N][N]
								)
{
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

	const int m = M, n = N;
	//double pythag(double a, double b);
	int flag,i,its,j,jj,k,l,nm;
	double anorm,c,f,g,h,s,scale,x,y,z;

	double rv1[n];// = std::vector<double>(n);
	g=scale=anorm=0.0;
	for (i=1;i<=n;i++) {
		l=i+1;
		rv1[i-1]=scale*g;
		g=s=scale=0.0;
		if (i <= m) {
			for (k=i;k<=m;k++) scale += fabs(mat[k-1][i-1]);
			if (scale) {
				for (k=i;k<=m;k++) {
					mat[k-1][i-1] /= scale;
					s += mat[k-1][i-1]*mat[k-1][i-1];
				}
				f=mat[i-1][i-1];
				g = -SIGN(sqrt(s),f);
				h=f*g-s;
				mat[i-1][i-1]=f-g;
				for (j=l;j<=n;j++) {
					for (s=0.0,k=i;k<=m;k++) s += mat[k-1][i-1]*mat[k-1][j-1];
					f=s/h;
					for (k=i;k<=m;k++) mat[k-1][j-1] += f*mat[k-1][i-1];
				}
				for (k=i;k<=m;k++) mat[k-1][i-1] *= scale;
			}
		}
		w[i-1]=scale *g;
		g=s=scale=0.0;
		if (i <= m && i != n) {
			for (k=l;k<=n;k++) scale += fabs(mat[i-1][k-1]);
			if (scale) {
				for (k=l;k<=n;k++) {
					mat[i-1][k-1] /= scale;
					s += mat[i-1][k-1]*mat[i-1][k-1];
				}
				f=mat[i-1][l-1];
				g = -SIGN(sqrt(s),f);
				h=f*g-s;
				mat[i-1][l-1]=f-g;
				for (k=l;k<=n;k++) rv1[k-1]=mat[i-1][k-1]/h;
				for (j=l;j<=m;j++) {
					for (s=0.0,k=l;k<=n;k++) s += mat[j-1][k-1]*mat[i-1][k-1];
					for (k=l;k<=n;k++) mat[j-1][k-1] += s*rv1[k-1];
				}
				for (k=l;k<=n;k++) mat[i-1][k-1] *= scale;
			}
		}
		anorm=_svd_max(anorm,(fabs(w[i-1])+fabs(rv1[i-1])));
	}
	for (i=n;i>=1;i--) {
		if (i < n) {
			if (g) {
				for (j=l;j<=n;j++)
					v[j-1][i-1]=(mat[i-1][j-1]/mat[i-1][l-1])/g;
				for (j=l;j<=n;j++) {
					for (s=0.0,k=l;k<=n;k++) s += mat[i-1][k-1]*v[k-1][j-1];
					for (k=l;k<=n;k++) v[k-1][j-1] += s*v[k-1][i-1];
				}
			}
			for (j=l;j<=n;j++) v[i-1][j-1]=v[j-1][i-1]=0.0;
		}
		v[i-1][i-1]=1.0;
		g=rv1[i-1];
		l=i;
	}
	for (i=_svd_min(m,n);i>=1;i--) {
		l=i+1;
		g=w[i-1];
		for (j=l;j<=n;j++) mat[i-1][j-1]=0.0;
		if (g) {
			g=1.0/g;
			for (j=l;j<=n;j++) {
				for (s=0.0,k=l;k<=m;k++) s += mat[k-1][i-1]*mat[k-1][j-1];
				f=(s/mat[i-1][i-1])*g;
				for (k=i;k<=m;k++) mat[k-1][j-1] += f*mat[k-1][i-1];
			}
			for (j=i;j<=m;j++) mat[j-1][i-1] *= g;
		} else for (j=i;j<=m;j++) mat[j-1][i-1]=0.0;
		++mat[i-1][i-1];
	}
	for (k=n;k>=1;k--) {
		for (its=1;its<=30;its++) {
			flag=1;
			for (l=k;l>=1;l--) {
				nm=l-1;
				if ((double)(fabs(rv1[l-1])+anorm) == anorm) {
					flag=0;
					break;
				}
				if ((double)(fabs(w[nm-1])+anorm) == anorm) break;
			}
			if (flag) {
				c=0.0;
				s=1.0;
				for (i=l;i<=k;i++) {
					f=s*rv1[i-1];
					rv1[i-1]=c*rv1[i-1];
					if ((double)(fabs(f)+anorm) == anorm) break;
					g=w[i-1];
					h=pythag(f,g);
					w[i-1]=h;
					h=1.0/h;
					c=g*h;
					s = -f*h;
					for (j=1;j<=m;j++) {
						y=mat[j-1][nm-1];
						z=mat[j-1][i-1];
						mat[j-1][nm-1]=y*c+z*s;
						mat[j-1][i-1]=z*c-y*s;
					}
				}
			}
			z=w[k-1];
			if (l == k) {
				if (z < 0.0) {
					w[k-1] = -z;
					for (j=1;j<=n;j++) v[j-1][k-1] = -v[j-1][k-1];
				}
				break;
			}
			//if (its == 30) nrerror("no convergence in 30 svdcmp iterations");
			x=w[l-1];
			nm=k-1;
			y=w[nm-1];
			g=rv1[nm-1];
			h=rv1[k-1];
			f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
			g=pythag(f,1.0);
			f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
			c=s=1.0;
			for (j=l;j<=nm;j++) {
				i=j+1;
				g=rv1[i-1];
				y=w[i-1];
				h=s*g;
				g=c*g;
				z=pythag(f,h);
				rv1[j-1]=z;
				c=f/z;
				s=h/z;
				f=x*c+g*s;
				g = g*c-x*s;
				h=y*s;
				y *= c;
				for (jj=1;jj<=n;jj++) {
					x=v[jj-1][j-1];
					z=v[jj-1][i-1];
					v[jj-1][j-1]=x*c+z*s;
					v[jj-1][i-1]=z*c-x*s;
				}
				z=pythag(f,h);
				w[j-1]=z;
				if (z) {
					z=1.0/z;
					c=f*z;
					s=h*z;
				}
				f=c*g+s*y;
				x=c*y-s*g;
				for (jj=1;jj<=m;jj++) {
					y=mat[jj-1][j-1];
					z=mat[jj-1][i-1];
					mat[jj-1][j-1]=y*c+z*s;
					mat[jj-1][i-1]=z*c-y*s;
				}
			}
			rv1[l-1]=0.0;
			rv1[k-1]=f;
			w[k-1]=x;
		}
	}
}
