#include "./NURBS.h"


py:: array_t<double> NURBSinterpolation1d(double xi, int p, py::array_t<double> in_knot, py::array_t<double> in_points, py::array_t<double> in_weights)
{
	/*	
	    xi   = point where we want to interpolate
		p:   = nurbs order
		knot = knot vector
		vector of points in format [pt1 pt2 pt3 pt4 .... ptn] 
	*/

	/* First get the inputs */
	py::buffer_info knot_buffer = in_knot.request();
	double *knot = static_cast<double *>(knot_buffer.ptr);
	int    numKnot = knot_buffer.size - 1;
	int    n       = numKnot - p - 1;
	
	py::buffer_info points_buffer = in_points.request();
	double *points = static_cast<double *>(points_buffer.ptr);
	
	py::buffer_info weights_buffer = in_weights.request();
	double *weight = static_cast<double *>(weights_buffer.ptr);
	
	if(fabs(xi-knot[numKnot-1]) < TOL) 
		xi = knot[numKnot-1] - TOL;
	
	/* and evaluate the non-zero basis functions*/
	
	double *N           = init1DArray(p+1);
	
	int span =    (p + n + 1) / 2;
	FindSpan      (n, p, xi, knot, &span); 
	BasisFuns     (span, xi, p, knot, N);
	
	/* and create NURBS approximation */
	double w_interp = 0.0, w=0.0;
    int c;
	for(c = 0; c <= p; c++)
	{
		w_interp      += N[c] * weight[span-p+c] * points[span-p+c];
		w             += N[c] * weight[span-p+c];
	}
	
	free(N);

	auto result = py::array_t<double>(1);
    py::buffer_info buf = result.request();
	double *ptr = static_cast<double *>(buf.ptr);
	
	ptr[0] = w_interp/w; 

	return result;
}


py:: array_t<double> NURBSinterpolation2d(double xi, double eta, int p, int q, py::array_t<double> knotU, py::array_t<double> knotV, py::array_t<double> in_pointsX, py::array_t<double> in_weights)
{
	/*	
	// We expect the function to be called as 
	// [interp] = NURBSinterpolation(xi, eta, p, q, uknot, vknot, pointsX, pointsY, weights)
	//	xi   = point where we want to interpolate [xi eta]
	//	uknot = knot vector u direction
    //	vknot = knot vector v direction
	//	vector of points in format [pt1 pt2 pt3 pt4 .... ptn] 
    */
	
	/* First get the inputs */
	
    py::buffer_info knotU_buffer = knotU.request();
	double *uknot = static_cast<double *>(knotU_buffer.ptr);
    py::buffer_info knotV_buffer = knotV.request();
	double *vknot = static_cast<double *>(knotV_buffer.ptr);

    py::buffer_info pointsX_buffer = in_pointsX.request();
	double *pointsX = static_cast<double *>(pointsX_buffer.ptr);
    
    py::buffer_info weights_buffer = in_weights.request();
	double *weight = static_cast<double *>(weights_buffer.ptr);
    
    /* create output */
    auto result = py::array_t<double>(1);
    py::buffer_info buf = result.request();
	double *interp = static_cast<double *>(buf.ptr);
    
    /* Then use them as in standard C*/
	int    numKnotU = knotU_buffer.size;
    int    numKnotV = knotV_buffer.size;
    
	int    mu       = numKnotU - 1;
    int    mv       = numKnotV - 1;
    
	int    nu       = mu - p - 1;
    int    nv       = mv - q - 1;
    
	
	if(fabs(xi -uknot[numKnotU-1]) < TOL) xi  = uknot[numKnotU-1] - TOL;
    if(fabs(eta-vknot[numKnotV-1]) < TOL) eta = vknot[numKnotV-1] - TOL;
	
	/* and evaluate the non-zero B-spline basis functions*/
	
	double *N        = init1DArray(p+1);
    double *M        = init1DArray(q+1);
	
	int spanU = (p + nu + 1) / 2; 
    int spanV = (q + nv + 1) / 2; 
    
    FindSpan      (nu, p, xi, uknot, &spanU);
    FindSpan      (nv, q, eta, vknot, &spanV);

	BasisFuns     (spanU, xi,  p, uknot, N);
    BasisFuns     (spanV, eta, q, vknot, M);
    
    /* and compute the approximation */
    
	interp[0] = 0.0;
    
    double tempu = 0, tempw = 0;
	
    int    vind, uind = spanU - p;
    int    k,l,id;

	for(l = 0; l <= q; l++)
    {
        vind = spanV - q +l;
        for(k = 0; k <= p; k++)
	    {
           id      = uind+k + vind*(nu+1); 
           tempu  += N[k] * pointsX[id] * weight[id] * M[l];
           tempw  += N[k] * weight[id] * M[l];
	    }
    }

    // projection
    interp[0] = tempu/tempw;
	
	free(N);
    free(M);

    return result;
}


py:: array_t<double> NURBSinterpolation3d(double xi, double eta, double zeta, int p, int q, int r, py::array_t<double> knotU, py::array_t<double> knotV, py::array_t<double> knotW, py::array_t<double> in_pointsX, py::array_t<double> in_weights)
{
	/*	
	// We expect the function to be called as 
	// [interp] = NURBSinterpolation(xi, eta, p, q, uknot, vknot, pointsX, pointsY, weights)
	//	xi   = point where we want to interpolate [xi eta]
	//	uknot = knot vector u direction
    //	vknot = knot vector v direction
	//	vector of points in format [pt1 pt2 pt3 pt4 .... ptn] 
    */
	
	/* First get the inputs */
	
    py::buffer_info knotU_buffer = knotU.request();
	double *uknot = static_cast<double *>(knotU_buffer.ptr);
    py::buffer_info knotV_buffer = knotV.request();
	double *vknot = static_cast<double *>(knotV_buffer.ptr);
    py::buffer_info knotW_buffer = knotW.request();
	double *wknot = static_cast<double *>(knotW_buffer.ptr);

    py::buffer_info pointsX_buffer = in_pointsX.request();
	double *pointsX = static_cast<double *>(pointsX_buffer.ptr);
    
    py::buffer_info weights_buffer = in_weights.request();
	double *weight = static_cast<double *>(weights_buffer.ptr);
    
    /* create output */
    auto result = py::array_t<double>(1);
    py::buffer_info buf = result.request();
	double *interp = static_cast<double *>(buf.ptr);
    
    /* Then use them as in standard C*/
	int    numKnotU = knotU_buffer.size;
    int    numKnotV = knotV_buffer.size;
    int    numKnotW = knotW_buffer.size;
    
	int    nu       = knotU_buffer.size - 1 - p - 1;
    int    nv       = knotV_buffer.size - 1 - q - 1;
    int    nw       = knotW_buffer.size - 1 - r - 1;
    
	if(fabs(xi -uknot[numKnotU-1]) < TOL) xi  = uknot[numKnotU-1] - TOL;
    if(fabs(eta-vknot[numKnotV-1]) < TOL) eta = vknot[numKnotV-1] - TOL;
    if(fabs(zeta-wknot[numKnotW-1]) < TOL) zeta = wknot[numKnotW-1] - TOL;
	
	/* and evaluate the non-zero B-spline basis functions*/
	
	double *N        = init1DArray(p+1);
    double *M        = init1DArray(q+1);
    double *P      = init1DArray(r+1);
	
	int spanU = (p + nu + 1) / 2; 
    int spanV = (q + nv + 1) / 2; 
    int spanW = (r + nw + 1) / 2;
    
    FindSpan      (nu, p, xi, uknot, &spanU);
    FindSpan      (nv, q, eta, vknot, &spanV);
    FindSpan      (nw, r, zeta, wknot, &spanW);

	BasisFuns     (spanU, xi,  p, uknot, N);
    BasisFuns     (spanV, eta, q, vknot, M);
    BasisFuns     (spanW, zeta, r, wknot, P);
    
    /* and compute the approximation */
    
	interp[0] = 0.0;
    
    double tempu = 0, tempw = 0;
	
    int    vind, wind, uind = spanU - p;
    int    k,i,j,id;

	for(k = 0; k <= r; k++)
    {
        wind = spanW - r + k;
        
        for(j = 0; j <= q; j++)
        {
            vind = spanV - q + j;
            
            for(i = 0; i <= p; i++)
            {
                id      = uind + i + (nu+1) * ( (nv+1)*wind + vind);
                tempu  += N[i] * pointsX[id] * weight[id] * M[j] * P[k];
                tempw  += N[i] * weight[id] * M[j] * P[k];
	        }
        }
    }

    // projection
    interp[0] = tempu/tempw;

	free(N);
    free(M);

    return result;
}


py:: array_t<double> NURBSinterpolationDers1d(double xi, int p, py::array_t<double> in_knot, py::array_t<double> in_points, py::array_t<double> in_weights)
{
	/*	
	    xi   = point where we want to interpolate
		p:   = nurbs order
		knot = knot vector
		vector of points in format [pt1 pt2 pt3 pt4 .... ptn] 
	*/

	/* First get the inputs */
	py::buffer_info knot_buffer = in_knot.request();
	double *knot = static_cast<double *>(knot_buffer.ptr);
	int    numKnot = knot_buffer.size;
	int    m       = numKnot - 1;
	int    n       = m - p -1;
	
	py::buffer_info points_buffer = in_points.request();
	double *points = static_cast<double *>(points_buffer.ptr);
	
	py::buffer_info weights_buffer = in_weights.request();
	double *weight = static_cast<double *>(weights_buffer.ptr);
	
	if(fabs(xi-knot[numKnot-1]) < TOL) 
		xi = knot[numKnot-1] - TOL;
	
	/* and evaluate the non-zero basis functions*/
	
	double *N           = init1DArray(p+1);
	double *NURBS       = init1DArray(p+1);
	double *NURBS_deriv = init1DArray(p+1);
	double **ders = init2DArray(n+1, p+1);
	
	int span = (p + n + 1) / 2;
	FindSpan      (n, p, xi, knot, &span); 
	BasisFuns     (span, xi, p, knot, N);
	dersBasisFuns (span, xi, p, n, knot, ders);	
	
	/* and create NURBS approximation */
	int k, c;
	
	for(k = 0; k <=p; k++)
	{
		double w_interp = 0.0, dw_interp_dxi= 0.0;
		
		for(c = 0; c <= p; c++)
		{
			w_interp      += N[c] * weight[span-p+c];
			dw_interp_dxi += ders[1][c] * weight[span-p+c];
		}

		NURBS[k]       = N[k] * weight[span-p+k] / w_interp;
		NURBS_deriv[k] = weight[span-p+k] * ( w_interp * ders[1][k] - dw_interp_dxi * N[k] ) / (w_interp * w_interp);
	}

	double interp = 0.0;
	double interp_deriv = 0.0;
	
	for(c = 0; c <= p; c++)
	{
		interp       += NURBS[c] * points[span-p+c];
		interp_deriv += NURBS_deriv[c] * points[span-p+c];
	}
	
	free(N);
	free(NURBS);
	free(NURBS_deriv);
	free2Darray(ders, (n+1));

	auto result = py::array_t<double>(2);
    py::buffer_info buf = result.request();
	double *ptr = static_cast<double *>(buf.ptr);
	
	ptr[0] = interp; 
	ptr[1] = interp_deriv;

	return result;
}


PYBIND11_MODULE(nurbs_interpolation, m)
{
    m.def("nurbs_interpolation1d", &NURBSinterpolation1d, "Return the NURBS basis function to Python");
	m.def("nurbs_interpolation2d", &NURBSinterpolation2d, "Return the NURBS basis function to Python");
	m.def("nurbs_interpolation3d", &NURBSinterpolation3d, "Return the NURBS basis function to Python");
	m.def("nurbs_interpolation_ders1d", &NURBSinterpolationDers1d, "Return the NURBS basis function to Python");
}
 
 
 

