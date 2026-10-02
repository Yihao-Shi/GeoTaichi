#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#define TOL 100*DBL_EPSILON

using namespace std;
namespace py = pybind11;


void     FindSpan(int n, int p, double u, double* U, int *mid);
void     BasisFuns( int i, double u, int p, double* U, double* N);
void     dersBasisFuns(int i, double u, int p, int order, double *knot, double **ders);
void     OneBasisFun(int p, int m, double *U, int i, double u, double *Nip);
void     dersOneBasisFuns(int p, int m, double *U, int i, double u, int n, double* ders);
double*  init1DArray(unsigned int x);
double** init2DArray(int x, int y);
void     free2Darray(double **array, int x);
void     print1DArray(double *array, int n);
void     print2DArray(double **array, int n, int m);
void     copy1DArray(int start, int end, double* ouput, double* input);
void     copy2DArray(int nstart, int nend, int mstart, int mend, double** ouput, double** input);


void copy1DArray(int start, int end, double* output, double* input)
{
	for(int i = start; i < end; i++)
	{
		output[i] = input[i];
	}
}

void copy2DArray(int nstart, int nend, int mstart, int mend, double** output, double** input)
{
	for(int i = nstart; i < nend; i++)
	{
        for(int j = mstart; j < mend; j++)
		{
            output[i][j] = input[i][j];
		}
	}
}

void print1DArray(double *array, int n)
{
	for(int i = 0; i < n; i++)
	{
		cout<<array[i]<<','<<' ';
	}
	cout<<endl;
}

void print2DArray(double **array, int n, int m)
{
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			cout<<array[i][j]<<','<<' ';
		}
		cout<<';'<<endl;
	}
}

void FindSpan(int n, int p, double u, double* U, int *mid)
{
    if( u>=U[n+1] )
	{
	    *mid = n;
        return;
	}
    if( u<=U[p] )
	{
	    *mid = p;
        return;
	}
 
    int low =p, high=n+1;
    while( u<U[*mid] || u>=U[*mid+1] )
    {
        if( u<U[*mid] )
            high=*mid;
        else
            low=*mid;
        *mid=(low+high)/2;
    }
}
 
 
void BasisFuns( int i, double u, int p, double* U, double* N)
{
	N[0] = 1.0;
	
	int j,r;
	double *left  = init1DArray(p+1);
	double *right = init1DArray(p+1);
	double saved, temp;
	
	for( j = 1; j <= p; ++j)
	{
		left[j]  = u - U[i+1-j];
		right[j] = U[i+j] - u;
		saved = 0.0;
		for(r = 0; r < j; ++r)
		{
			temp  = N[r] / ( right[r+1] + left[j-r] );
			N[r]  = saved + right[r+1] * temp;
			saved = left[j-r] * temp;
		}
		N[j]= saved;
	}
	
	free(left);
	free(right);
	
}
 
void dersBasisFuns(int i, double u, int p, int order, double *knot, double **ders)
{
   double saved,temp;
   int j,k,j1,j2,r;
  
   double *left  = init1DArray(p+1);
   double *right = init1DArray(p+1);
	
   double **ndu  = init2DArray(p+1, p+1);
   double **a    = init2DArray(p+1, p+1);
   
   ndu[0][0]=1.;
   for( j=1; j<=p; j++ )
   {
     left[j]=u-knot[i+1-j];
     right[j]=knot[i+j]-u;
     saved=0.0;
     for( r=0; r<j; r++ )
     {
       ndu[j][r]=right[r+1]+left[j-r];
       temp=ndu[r][j-1]/ndu[j][r];
      
       ndu[r][j]=saved+right[r+1]*temp;
       saved=left[j-r]*temp;
     }
     ndu[j][j]=saved;
   }
   for( j=0; j<=p; j++ )
     ders[0][j]=ndu[j][p];  
    
   if( order==0 )
     return;
  

   for( r=0; r<=p; r++ )
   {
     int s1=0, s2=1;   
     a[0][0]=1.0;

     for( k=1; k<=order; k++ )
     {
       double d=0.;
       int rk=r-k, pk=p-k;
       if( r>=k )
       {
 			a[s2][0]=a[s1][0]/ndu[pk+1][rk];
 			d=a[s2][0]*ndu[rk][pk];
       }
       j1 = rk >= -1 ? 1 : -rk;
       j2 = (r-1<=pk) ? k-1 : p-r;
       for( j=j1; j<=j2; j++ )
       {
           a[s2][j]=(a[s1][j]-a[s1][j-1])/ndu[pk+1][rk+j];
           d+=a[s2][j]*ndu[rk+j][pk];
       }
       if( r<=pk )
       {
           a[s2][k]= -a[s1][k-1]/ndu[pk+1][r];
           d+=a[s2][k]*ndu[r][pk];
       }
       ders[k][r]=d;
       j=s1; s1=s2; s2=j;  
     }
   }
   r=p;
   for( k=1; k<=order; k++ )
   {
     for( j=0; j<=p; j++ ) 
       ders[k][j]*=r;
     r*=(p-k);
   }
   
   free(left); 
   free(right);
   
   free2Darray(ndu, p+1);
   free2Darray(a, p+1);
    
 }


void OneBasisFun(int p, int m, double *U, int i, double u, double *Nip)
{
	double *N = init1DArray(p+1);
	
	if((i == 0 && u == U[0] ) ||
	   (i == (m-p-1) && u == U[m]))
	{
		*Nip = 1.0;
		return;
	}
	
	if(u < U[i] || u >= U[i+p+1])
	{
		*Nip = 0.0;
		return;
	}
	
	int j;
	for(j = 0; j <= p; j++)
	{
		if(u >= U[i+j] && u < U[i+j+1]) N[j] = 1.0;
		else N[j] = 0.0;
	}
	
	int k;
	double saved, Uleft, Uright, temp;
	
	for(k = 1; k <= p; k++)
	{
		if(N[0] == 0.0) saved = 0.0;
		else saved = ((u-U[i]) * N[0])/ (U[i+k] - U[i]);
		for(j = 0; j < (p-k+1); j++)
		{
			Uleft = U[i+j+1];
			Uright = U[i+j+k+1];
			if(N[j+1] == 0.0)
			{
				N[j] = saved; saved = 0.0;
			}
			else 
			{
				temp = N[j+1] / (Uright-Uleft);
				N[j] = saved + (Uright - u) * temp;
				saved = (u-Uleft) * temp;
			}
		}
	}
	
	*Nip = N[0];
	
	free(N);
}


void dersOneBasisFuns(int p, int m, double *U, int i, double u, int order, double* ders)
{
	double **N = init2DArray(order+1, order+1);
	double *ND = init1DArray(order+1);
	
	int k, j, jj;
	double Uleft, Uright, saved, temp;
	
	if(u < U[i] || u >= U[i+p+1])
	{
		for(k = 0; k <= order; k++)
		{
			ders[k] = 0.0;
		}
		return;

	}

	for(j = 0; j <= p; j++)
	{
		if(u >= U[i+j] && u < U[i+j+1])
			N[j][0] = 1.0;
		else
			N[j][0] = 0.0;
	}

	for(k = 1; k <= p; k++)
	{
		if(N[0][k-1]==0.0) 
			saved = 0.0;
		else
			saved = ((u - U[i]) * N[0][k-1]) / ( U[i+k] - U[i] );
		
		for(j = 0; j < (p-k+1); j++)
		{
			Uleft = U[i+j+1];
			Uright = U[i+j+k+1];
			if(N[j+1][k-1] == 0.0)
			{
				N[j][k] = saved; saved = 0.0;
			}
			else
			{
				temp = N[j+1][k-1] / (Uright - Uleft);
				N[j][k] = saved + (Uright - u) * temp;
				saved = (u - Uleft) * temp;
			}
		}	
	}
	
	ders[0] = N[0][p];
	
	for(k = 1; k<=order; k++)
	{
		for(j = 0; j <=k; j++)
			ND[j] = N[j][p-k];
        for(jj = 1; jj <=k; jj++)
        {
            if(ND[0] == 0.0) 
                saved = 0.0;
            else
                saved = ND[0] / ( U[i+p-k+jj] - U[i]);
            for(j = 0; j<(k-jj+1); j++)
            {
                Uleft = U[i+j+1];
                Uright = U[i+j+p+jj];
                if(ND[j+1] ==0.0)
                {
                    ND[j] = (p-k+jj) * saved; saved = 0.0;
                }
                else
                {
                    temp = ND[j+1] / (Uright - Uleft);
                    ND[j] = (p-k+jj) * (saved - temp);
                    saved = temp;
                }
            }
        }
        ders[k] = ND[0];
	}

	free2Darray(N, order+1);
	free(ND);
}


double* init1DArray(unsigned int x)
{
 	double *array = (double*)calloc(x, sizeof(double));
 	return array;
}


double** init2DArray(int x, int y)
{
 	double **array = (double **)calloc(x, sizeof(double *));
 	
 	int c;
 	for(c = 0; c < x; c++)
 	{
 		array[c] = (double*)calloc(y, sizeof(double));
 	}

 	return array;
}
 

void free2Darray(double **array, int x)
{
 	int c;
 	for(c = 0; c < x; c++)
 	{
 		free(array[c]);
 	}
 	free(array);
}

 

/*PYBIND11_MODULE(nurbs, m) {

    m.doc() = "Build basic nurbs functions via c backend\n";

    m.def("find_span", &FindSpan, 
	      R"pbdoc(This function determines the knot span.
    	   ie. if we have a coordinate u which lies in the range u \in [u_i, u_{i+1})
		   we want to find i
		   Note that: u_i <= u < (not equal) u_{i+1}!!!
		   If we have knot = [0,0.5,1] then u=0.5 has span=1 not 0!!!)pbdoc",
          py::arg("n"), py::arg("p"), py::arg("u"), py::arg("U")
         );

	m.def("basis_funs", &BasisFuns, 
	      R"pbdoc(we can compute the non zero basis functions
          at point u, there are p+1 non zero basis functions)pbdoc",
          py::arg("i"), py::arg("u"), py::arg("p"), py::arg("U"), py::arg("N")
         );

	m.def("ders_basis_funs", 
	      [](int i, double u, int p, int order, double* knot, std::vector<double *> ders)
		  {
		    return dersBasisFuns(i, u, p, order, knot, ders.data());
	      }, 
	      R"pbdoc(Calculate the non-zero derivatives of the b-spline functions)pbdoc",
          py::arg("i"), py::arg("u"), py::arg("p"), py::arg("order"), py::arg("knot"), py::arg("ders")
         );
	
	m.def("one_basis_fun", &OneBasisFun, 
	      R"pbdoc(Compute an individual B-spline function)pbdoc",
          py::arg("p"), py::arg("m"), py::arg("U"), py::arg("i"), py::arg("u")
         );

	m.def("ders_one_basis_fun", &dersOneBasisFuns, 
	      R"pbdoc(Compute the derivatives for basis function Nip)pbdoc",
          py::arg("p"), py::arg("m"), py::arg("U"), py::arg("i"), py::arg("u"), py::arg("n"), py::arg("ders")
         );
}*/