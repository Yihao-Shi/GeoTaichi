#include "./NURBS.h"


py:: array_t<double> BSPLINE2DBasisDers(double xi, double eta, int p, int q, py::array_t<double> knotU, py::array_t<double> knotV, py::array_t<double> in_weights)
{
	/* 
	// All non-zero basis functions and derivatives at point [xi,eta] are computed.
	// We expect the function to be called as 
	// [R dRdxi dRdeta] = NURBSinterpolation(xi, p, q, knotU, knotV, weights)
	// xi           = point, [xi eta], where we want to interpolate
	// knotU, knotV = knot vectors
	// weights      = vector of weights 
    */

	/* First get the inputs */
    py::buffer_info knotU_buffer = knotU.request();
	double *uknot = static_cast<double *>(knotU_buffer.ptr);
    py::buffer_info knotV_buffer = knotV.request();
	double *vknot = static_cast<double *>(knotV_buffer.ptr);

    int    numKnotU = knotU_buffer.size;
    int    numKnotV = knotV_buffer.size;
    
	int    nU       = numKnotU - 1 - p - 1;
    int    nV       = numKnotV - 1 - q - 1;
    int    noFuncs  = (p+1)*(q+1); 
	
	if(fabs(xi-uknot[numKnotU-1]) < TOL) 
		xi = uknot[numKnotU-1] - TOL;
    
	if(fabs(eta-vknot[numKnotV-1]) < TOL) 
		eta = vknot[numKnotV-1] - TOL; 
	
	/* and evaluate the non-zero univariate B-spline basis functions
     * and first derivatives 
     */
	
	double *N      = (double *)malloc(sizeof(double)*(p+1));
    double *M      = (double *)malloc(sizeof(double)*(q+1));
	double **dersN = init2DArray(nU+1, p+1);
    double **dersM = init2DArray(nV+1, q+1);
	
	int spanU = (p + nU + 1) / 2; 
    int spanV = (q + nV + 1) / 2; 
    
    FindSpan      (nU, p, xi, uknot, &spanU);
    FindSpan      (nV, q, eta, vknot, &spanV);
    
	BasisFuns     (spanU, xi, p, uknot, N);
    BasisFuns     (spanV, eta, q, vknot, M);
    
	dersBasisFuns (spanU, xi, p, nU, uknot, dersN);	
    dersBasisFuns (spanV, eta, q, nV, vknot, dersM);
    
	
	/* and create NURBS approximation */
    int i, j, k, c;
    int uind = spanU - p;
    int vind;
    
    double w     = 0.0;
    double dwdxi = 0.0;
    double dwdet = 0.0;
    double wgt;
	
    for(j = 0; j <= q; j++)
    {
        vind = spanV - q + j;  
        
        for(i = 0; i <= p; i++)
        {               
            c   = uind + i + vind * (nU+1);            
            wgt = weight[c];
            
            w     += N[i]        * M[j] * wgt;
            dwdxi += dersN[1][i] * M[j] * wgt;
            dwdet += dersM[1][j] * N[i] * wgt;
        }
    }
    
	/* create output */
    
    auto result = py::array_t<double>(3 * noFuncs);
    py::buffer_info buf = result.request();
	double *ptr = static_cast<double *>(buf.ptr);
    
    k    = 0;
    
    double fac;
    
    for(j = 0; j <= q; j++)
    {
        for(i = 0; i <= p; i++)
        {               
            ptr[3 * k]     = N[i]*M[j];
            ptr[3 * k + 1] = dersN[1][i]*M[j];
            ptr[3 * k + 2] = dersM[1][j]*N[i];
            k += 1;
        }
    }
                
	free(N);
    free(M);
	free2Darray(dersN, (nU+1));
    free2Darray(dersM, (nV+1));	

    return result;
}

PYBIND11_MODULE(bspline2d_basis_ders, m)
{
    m.def("bslpine2d_basis_ders", &BSPLINE2DBasisDers, "Return the 2D BSpline basis functions and first derivatives to Python");;
}
 
 
 
 

