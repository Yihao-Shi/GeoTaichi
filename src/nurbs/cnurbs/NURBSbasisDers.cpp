#include "./NURBS.h"


py:: array_t<double> NURBSbasis(int i, int p, double xi, py::array_t<double> in_knot, py::array_t<double> in_weights)
{
	/*	
	// p = order of basis (0,1,2 etc)
	// knot = knot vector
	// i = basis function we want (1,2,3 ie. Hughes' notation)
	// xi = the coordinate we wish to evaluate at
	// weights = the weightings for the NURBS function (length(weights) = m - p -1) 
	*/

	py::buffer_info knot_buffer = in_knot.request();
	double *knot = static_cast<double *>(knot_buffer.ptr);
	int    numKnot = knot_buffer.size;
	int    m       = numKnot - 1;

	py::buffer_info weights_buffer = in_weights.request();
	double *weight = static_cast<double *>(weights_buffer.ptr);
	
	if(fabs(xi-knot[numKnot-1]) < TOL) 
		xi = knot[numKnot-1] - TOL;
	
	/* and call the basis function routine*/
	
	double Rip, Nip, dRip_dxi;
	double w_interp, dw_interp_dxi ;
	
	int n = m - p -1;
	
	double *N     = init1DArray(p+1);
	double *dN    = init1DArray(p+1);
	double **ders = init2DArray(p+1, p+1);
	
	int span = (p + n + 1) / 2;
	FindSpan     (n, p, xi, knot, &span);
	BasisFuns    (span, xi, p, knot, N);
	dersBasisFuns(span, xi, p, p, knot, ders);	

	w_interp      = 0.0; 
	dw_interp_dxi = 0.0;
	
	int c;
	for(c = 0; c <= p; c++)
	{
	  w_interp      += N[c] * weight[span-p+c];
	  dw_interp_dxi += ders[1][c] * weight[span-p+c];
	}
	
	dersOneBasisFuns(p, m, knot, i, xi, p, dN);	

	OneBasisFun(p, m, knot, i, xi, &Nip);			
	
	Rip = Nip * weight[i] / w_interp;					
	
	dRip_dxi = weight[i] * ( w_interp * dN[1] - dw_interp_dxi * Nip ) / (w_interp * w_interp);


	free2Darray(ders, (p+1));
	free(N); free(dN);
	
	auto result = py::array_t<double>(2);
    py::buffer_info buf = result.request();
	double *ptr = static_cast<double *>(buf.ptr);
	
	ptr[0] = Rip; 
	ptr[1] = dRip_dxi;

	return result;
}

py:: array_t<double> NURBS1DBasisDers(double xi, int p, py::array_t<double> knotU, py::array_t<double> in_weights)
{
    /* 
    // All non-zero basis functions and derivatives at point [xi] are computed.
    //
    // We expect the function to be called as
    // [R dRdxi]      = NURBS1DBasisDers(xi,p,knotU,weights)
    //
    //	xi           = where we want to interpolate
    //	knotU        = knot vector
    //	weights      = vector of weights
    */

    /* First get the inputs */
    py::buffer_info knotU_buffer = knotU.request();
	double *uknot = static_cast<double *>(knotU_buffer.ptr);
    int    numKnotU = knotU_buffer.size;
    int    nU       = numKnotU - 1 - p - 1;
    int    noFuncs  = (p+1);
    py::buffer_info weights_buffer = in_weights.request();
	double *weight = static_cast<double *>(weights_buffer.ptr);
    
    if(fabs(xi-uknot[numKnotU-1]) < TOL)
        xi = uknot[numKnotU-1] - TOL;
    
    /* and evaluate the non-zero univariate B-spline basis functions
     * and first derivatives
     */
    
    double *N      = init1DArray(p+1);
    double **dersN = init2DArray(nU+1, p+1);

    int spanU = (p + nU + 1) / 2;
	FindSpan      (nU, p, xi, uknot, &spanU); 
    
    BasisFuns     (spanU, xi, p, uknot, N);
    dersBasisFuns (spanU, xi, p, nU, uknot, dersN);
    
    /* and create NURBS approximation */
    
    int i;
    int    uind  = spanU - p;
    double w     = 0.0;
    double dwdxi = 0.0;
    double wgt;
    
    for(i = 0; i <= p; i++)
    {       
        wgt    = weight[uind+i];        
        w     += N[i]        * wgt;
        dwdxi += dersN[1][i] * wgt;        
    }
        
    /* create output */
    auto result = py::array_t<double>(2 * noFuncs);
    py::buffer_info buf = result.request();
	double *ptr = static_cast<double *>(buf.ptr);
    
    double fac;
    for(i = 0; i <= p; i++)
    {
        fac      = weight[uind+i]/(w*w);
        
        ptr[2 * i]     = N[i]*fac*w;
        ptr[2 * i + 1] = (dersN[1][i]*w - N[i]*dwdxi) * fac;                       
    }
       
    free(N);
    free2Darray(dersN, (nU+1));

    return result;
}


py:: array_t<double> NURBS1DBasis2ndDers(double xi, int p, py::array_t<double> knotU, py::array_t<double> in_weights)
{
    /* 
    // All non-zero basis functions and derivatives at point [xi] arecomputed.
    // 
    // We expect the function to be called as
    // [R dRdxi dR2dxi]      = NURBS1DBasis2ndDers(xi,p,knotU,weights)
    // 
    // xi           = where we want to interpolate
    // knotU        = knot vector
    // weights      = vector of weights
    */

    /* First get the inputs */
    py::buffer_info knotU_buffer = knotU.request();
	double *uknot = static_cast<double *>(knotU_buffer.ptr);
    int    numKnotU = knotU_buffer.size;
    int    nU       = numKnotU - 1 - p - 1;
    int    noFuncs  = (p+1);
    py::buffer_info weights_buffer = in_weights.request();
	double *weight = static_cast<double *>(weights_buffer.ptr);
    
    if(fabs(xi-uknot[numKnotU-1]) < TOL)
        xi = uknot[numKnotU-1] - TOL;
    
    /* and evaluate the non-zero univariate B-spline basis functions
     * and first derivatives
     */
    
    double *N      = init1DArray(p+1);
    double **dersN = init2DArray(nU+1, p+1);
    
    int spanU = (p + nU + 1) / 2;
	FindSpan      (nU, p, xi, uknot, &spanU); 
    
    BasisFuns     (spanU, xi, p, uknot, N);
    dersBasisFuns (spanU, xi, p, nU, uknot, dersN);

    /* and create NURBS approximation */
    
    int i;
    int    uind   = spanU - p;
    double w      = 0.0; /* N_iw_i */
    double dwdxi  = 0.0; /* first derivative of w */
    double d2wdxi = 0.0; /* second derivative of w */
    double wi;           /* wi */
    
    for(i = 0; i <= p; i++)
    {       
        wi      = weight[uind+i];        
        w      += N[i]        * wi;
        dwdxi  += dersN[1][i] * wi;        
        d2wdxi += dersN[2][i] * wi;        
    }
        
    /* create output */
    auto result = py::array_t<double>(3 * noFuncs);
    py::buffer_info buf = result.request();
	double *ptr = static_cast<double *>(buf.ptr);
    
    double fac;
    for(i = 0; i <= p; i++)
    {
        wi        = weight[uind+i]; 
        fac       = wi/(w*w);
        ptr[3 * i]  = N[i]*wi/w;
        ptr[3 * i + 1]  = (dersN[1][i]*w - N[i]*dwdxi) * fac;  
        ptr[3 * i + 2] = wi*(dersN[2][i]/w - 2*dersN[1][i]*dwdxi/w/w - N[i]*d2wdxi/w/w + 2*N[i]*dwdxi*dwdxi/w/w/w) ;                       
    }
       
    free(N);
    free2Darray(dersN, (nU+1));

    return result;
}


py:: array_t<double> NURBS2DBasisDers(double xi, double eta, int p, int q, py::array_t<double> knotU, py::array_t<double> knotV, py::array_t<double> in_weights)
{
	/* 
	// All non-zero basis functions and derivatives at point [xi,eta] are
    // computed.
	//
	// We expect the function to be called as 
	// [R dRdxi dRdeta] = NURBSinterpolation(xi, p, q, knotU, knotV, weights)
	//
	//	xi           = point, [xi eta], where we want to interpolate
	//	knotU, knotV = knot vectors
	//	weights      = vector of weights 
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
	
	py::buffer_info weights_buffer = in_weights.request();
	double *weight = static_cast<double *>(weights_buffer.ptr);
	
	if(fabs(xi-uknot[numKnotU-1]) < TOL) 
		xi = uknot[numKnotU-1] - TOL;
    
	if(fabs(eta-vknot[numKnotV-1]) < TOL) 
		eta = vknot[numKnotV-1] - TOL; 
	
	/* and evaluate the non-zero univariate B-spline basis functions
     * and first derivatives 
     */
	
	double *N      = init1DArray(p+1);
    double *M      = init1DArray(q+1);
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
    
    uind = spanU - p;
    k    = 0;
    
    double fac;
    for(j = 0; j <= q; j++)
    {
        vind = spanV - q + j; 
        
        for(i = 0; i <= p; i++)
        {               
            c        = uind + i + vind*(nU+1);
            fac      = weight[c]/(w*w);
            
            ptr[3 * k]     = N[i]*M[j]*fac*w;
            ptr[3 * k + 1] = (dersN[1][i]*M[j]*w - N[i]*M[j]*dwdxi) * fac;
            ptr[3 * k + 2] = (dersM[1][j]*N[i]*w - N[i]*M[j]*dwdet) * fac;
            
            k += 1;
        }
    }
                
	free(N);
    free(M);
	free2Darray(dersN, (nU+1));
    free2Darray(dersM, (nV+1));	

    return result;
}


py:: array_t<double> NURBS2DBasis2ndDers(double xi, double eta, int p, int q, py::array_t<double> knotU, py::array_t<double> knotV, py::array_t<double> in_weights)
{
	/* 
	// All non-zero basis functions and derivatives at point [xi,eta] are computed.
	//
	// We expect the function to be called as 
	// [R dRdxi dRdeta dR2dxi dR2deta dR2dxideta] = NURBS2DBasis2ndDers(xi, p, q, knotU,knotV, weights)
	//
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
	
	py::buffer_info weights_buffer = in_weights.request();
	double *weight = static_cast<double *>(weights_buffer.ptr);
	
	if(fabs(xi-uknot[numKnotU-1]) < TOL) 
		xi = uknot[numKnotU-1] - TOL;
    
	if(fabs(eta-vknot[numKnotV-1]) < TOL) 
		eta = vknot[numKnotV-1] - TOL; 
	
	/* and evaluate the non-zero univariate B-spline basis functions
     * and first derivatives 
     */
	
	double *N      = init1DArray(p+1);
    double *M      = init1DArray(q+1);
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
    
    double w      = 0.0; /* w = N_I w_I*/
    double dwdxi  = 0.0; /* first derivative of w w.r.t xi*/
    double d2wdxi = 0.0; /* second derivative of w w.r.t xi*/
    double dwdet  = 0.0; /* first derivative of w w.r.t eta*/
    double d2wdet = 0.0; /* second derivative of w w.r.t eta*/
    double d2wdxe = 0.0; /* second derivative of w w.r.t xi-eta*/
    double wi;
	
    for(j = 0; j <= q; j++)
    {
        vind = spanV - q + j;  
        
        for(i = 0; i <= p; i++)
        {               
            c   = uind + i + vind * (nU+1);            
            wi  = weight[c];
            
            w      += N[i]        * M[j] * wi;
            dwdxi  += dersN[1][i] * M[j] * wi;
            d2wdxi += dersN[2][i] * M[j] * wi;
            dwdet  += dersM[1][j] * N[i] * wi;
            d2wdet += dersM[2][j] * N[i] * wi;
            d2wdxe += dersN[1][i] * dersM[1][j] * wi;
        }
    }

	/* create output */
    auto result = py::array_t<double>(6 * noFuncs); /* R dRdxi dRdeta d2Rdxi2 d2Rdeta2 d2Rdxideta*/
    py::buffer_info buf = result.request();
	double *ptr = static_cast<double *>(buf.ptr);
    
    uind = spanU - p;
    k    = 0;
    
    double fac;
    for(j = 0; j <= q; j++)
    {
        vind = spanV - q + j; 
        
        for(i = 0; i <= p; i++)
        {               
            c        = uind + i + vind*(nU+1);
            fac      = weight[c]/(w*w);
            wi       = weight[c];
            
            ptr[6 * k]     = N[i]*M[j]*fac*w;
            
            ptr[6 * k] = (dersN[1][i]*M[j]*w - N[i]*M[j]*dwdxi) * fac;
            ptr[6 * k + 1] = (dersM[1][j]*N[i]*w - N[i]*M[j]*dwdet) * fac;
            
            ptr[6 * k + 2] = wi*(dersN[2][i]*M[j]/w - 2*dersN[1][i]*M[j]*dwdxi/w/w - N[i]*M[j]*d2wdxi/w/w + 2*N[i]*M[j]*dwdxi*dwdxi/w/w/w);
            ptr[6 * k + 3] = wi*(dersM[2][j]*N[i]/w - 2*dersM[1][j]*N[i]*dwdet/w/w - N[i]*M[j]*d2wdet/w/w + 2*N[i]*M[j]*dwdet*dwdet/w/w/w);
            ptr[6 * k + 4] = wi*(dersN[1][i]*dersM[1][j]/w - dersN[1][i]*M[j]*dwdet/w/w - N[i]*dersM[1][j]*dwdxi/w/w - N[i]*M[j]*d2wdxe/w/w + 2*N[i]*M[j]*dwdxi*dwdet/w/w/w);
            
            k += 1;
        }
    }

	free(N);
    free(M);
	free2Darray(dersN, (nU+1));
    free2Darray(dersM, (nV+1));	

    return result;
}


py:: array_t<double> NURBS3DBasisDers(double xi, double eta, double zeta, int p, int q, int r, py::array_t<double> knotU, py::array_t<double> knotV, py::array_t<double> knotW, py::array_t<double> in_weights)
{
    /*  
    // All non-zero basis functions and derivatives at point [xi,eta,zeta] are computed.
    
    // We expect the function to be called as
    // [R dRdxi dRdeta dRdzeta] = NURBS3DBasisDers(xi, p, q, r, knotU, knotV, knotW, weights)
    
    // xi                  = point, [xi eta zeta], where we want to interpolate
    // knotU, knotV, knotZ = knot vectors
    // weights             = vector of weights
    */
    
    /* First get the inputs */
    py::buffer_info knotU_buffer = knotU.request();
	double *uknot = static_cast<double *>(knotU_buffer.ptr);
    py::buffer_info knotV_buffer = knotV.request();
	double *vknot = static_cast<double *>(knotV_buffer.ptr);
    py::buffer_info knotW_buffer = knotW.request();
	double *wknot = static_cast<double *>(knotW_buffer.ptr);

    int    numKnotU = knotU_buffer.size;
    int    numKnotV = knotV_buffer.size;
    int    numKnotW = knotW_buffer.size;
    
    int    nU       = numKnotU - 1 - p - 1;
    int    nV       = numKnotV - 1 - q - 1;
    int    nW       = numKnotW - 1 - r - 1;
    int    noFuncs  = (p+1)*(q+1)*(r+1);
    
    py::buffer_info weights_buffer = in_weights.request();
	double *weight = static_cast<double *>(weights_buffer.ptr);
    
    if(fabs(xi-uknot[numKnotU-1]) < TOL)
        xi = uknot[numKnotU-1] - TOL;
    
    if(fabs(eta-vknot[numKnotV-1]) < TOL)
        eta = vknot[numKnotV-1] - TOL;
    
    if(fabs(zeta-wknot[numKnotW-1]) < TOL)
        zeta = wknot[numKnotW-1] - TOL;
    
    /* and evaluate the non-zero univariate B-spline basis functions
     * and first derivatives
     */
    
    double *N      = init1DArray(p+1);
    double *M      = init1DArray(q+1);
    double *P      = init1DArray(r+1);
    
    double **dersN = init2DArray(nU+1, p+1);
    double **dersM = init2DArray(nV+1, q+1);
    double **dersP = init2DArray(nW+1, r+1);
    
    int spanU = (p + nU + 1) / 2;
    int spanV = (q + nV + 1) / 2;
    int spanW = (r + nW + 1) / 2;
    
    FindSpan      (nU, p, xi, uknot, &spanU);
    FindSpan      (nV, q, eta, vknot, &spanV);
    FindSpan      (nW, r, zeta, wknot, &spanW);

    BasisFuns     (spanU, xi, p, uknot, N);
    BasisFuns     (spanV, eta, q, vknot, M);
    BasisFuns     (spanW, zeta, r, wknot, P);
    
    dersBasisFuns (spanU, xi, p, nU, uknot, dersN);
    dersBasisFuns (spanV, eta, q, nV, vknot, dersM);
    dersBasisFuns (spanW, zeta, r, nW, wknot, dersP);
    
    /* and create NURBS approximation */
    
    int i, j, k, c, kk;
    int uind = spanU - p;
    int vind, wind;
    
    double w     = 0.0;
    double dwdxi = 0.0;
    double dwdet = 0.0;
    double dwdze = 0.0;
    double wgt;
    
    for(k = 0; k <= r; k++)
    {
        wind = spanW - r + k;
        
        for(j = 0; j <= q; j++)
        {
            vind = spanV - q + j;
            
            for(i = 0; i <= p; i++)
            {
                c   = uind + i + (nU+1) * ( (nV+1)*wind + vind);
                wgt = weight[c];
                
                w     += N[i]        * M[j] * P[k] * wgt;
                dwdxi += dersN[1][i] * M[j] * P[k] * wgt;
                dwdet += dersM[1][j] * N[i] * P[k] * wgt;
                dwdze += dersP[1][k] * N[i] * M[j] * wgt;
            }
        }
    }
    
    /* create output */
    auto result = py::array_t<double>(4 * noFuncs);
    py::buffer_info buf = result.request();
	double *ptr = static_cast<double *>(buf.ptr);
    
    uind = spanU - p;
    kk   = 0;
    
    double fac;
    double nmp;
    
    for(k = 0; k <= r; k++)
    {
        wind = spanW - r + k;
        
        for(j = 0; j <= q; j++)
        {
            vind = spanV - q + j;
            
            for(i = 0; i <= p; i++)
            {
                c        = uind + i + (nU+1) * ( (nV+1)*wind + vind);
                fac      = weight[c]/(w*w);
                nmp      = N[i]*M[j]*P[k];
                
                ptr[4 * kk] = nmp * fac * w;
                ptr[4 * kk + 1] = (dersN[1][i]*M[j]*P[k]*w - nmp*dwdxi) * fac;
                ptr[4 * kk + 2] = (dersM[1][j]*N[i]*P[k]*w - nmp*dwdet) * fac;
                ptr[4 * kk + 3] = (dersP[1][k]*N[i]*M[j]*w - nmp*dwdze) * fac;
                
                kk      += 1;
            }
        }
    }

    free(N);
    free(M);
    free(P);
    free2Darray(dersN, (nU+1));
    free2Darray(dersM, (nV+1));
    free2Darray(dersP, (nW+1));

    return result;
}


py:: array_t<double> NURBS3DBasis2ndDers(double xi, double eta, double zeta, int p, int q, int r, py::array_t<double> knotU, py::array_t<double> knotV, py::array_t<double> knotW, py::array_t<double> in_weights)
{
    /*  
    // All non-zero basis functions and derivatives at point [xi,eta,zeta] are computed.
    
    // We expect the function to be called as
    // [R dRdxi dRdeta dRdzeta] = NURBS3DBasisDers(xi, p, q, r, knotU, knotV, knotW, weights)
    
    // xi                  = point, [xi eta zeta], where we want to interpolate
    // knotU, knotV, knotZ = knot vectors
    // weights             = vector of weights
    */
    
    /* First get the inputs */
    py::buffer_info knotU_buffer = knotU.request();
	double *uknot = static_cast<double *>(knotU_buffer.ptr);
    py::buffer_info knotV_buffer = knotV.request();
	double *vknot = static_cast<double *>(knotV_buffer.ptr);
    py::buffer_info knotW_buffer = knotW.request();
	double *wknot = static_cast<double *>(knotW_buffer.ptr);

    int    numKnotU = knotU_buffer.size;
    int    numKnotV = knotV_buffer.size;
    int    numKnotW = knotW_buffer.size;
    
    int    nU       = numKnotU - 1 - p - 1;
    int    nV       = numKnotV - 1 - q - 1;
    int    nW       = numKnotW - 1 - r - 1;
    int    noFuncs  = (p+1)*(q+1)*(r+1);
    
    py::buffer_info weights_buffer = in_weights.request();
	double *weight = static_cast<double *>(weights_buffer.ptr);
    
    if(fabs(xi-uknot[numKnotU-1]) < TOL)
        xi = uknot[numKnotU-1] - TOL;
    
    if(fabs(eta-vknot[numKnotV-1]) < TOL)
        eta = vknot[numKnotV-1] - TOL;
    
    if(fabs(zeta-wknot[numKnotW-1]) < TOL)
        zeta = wknot[numKnotW-1] - TOL;
    
    /* and evaluate the non-zero univariate B-spline basis functions
     * and first derivatives
     */
    
    double *N      = init1DArray(p+1);
    double *M      = init1DArray(q+1);
    double *P      = init1DArray(r+1);
    
    double **dersN = init2DArray(nU+1, p+1);
    double **dersM = init2DArray(nV+1, q+1);
    double **dersP = init2DArray(nW+1, r+1);
    
    int spanU = (p + nU + 1) / 2;
    int spanV = (q + nV + 1) / 2;
    int spanW = (r + nW + 1) / 2;
    
    FindSpan      (nU, p, xi, uknot, &spanU);
    FindSpan      (nV, q, eta, vknot, &spanV);
    FindSpan      (nW, r, zeta, wknot, &spanW);

    BasisFuns     (spanU, xi, p, uknot, N);
    BasisFuns     (spanV, eta, q, vknot, M);
    BasisFuns     (spanW, zeta, r, wknot, P);
    
    dersBasisFuns (spanU, xi, p, nU, uknot, dersN);
    dersBasisFuns (spanV, eta, q, nV, vknot, dersM);
    dersBasisFuns (spanW, zeta, r, nW, wknot, dersP);
    
    /* and create NURBS approximation */
    
    int i, j, k, c, kk;
    int uind = spanU - p;
    int vind, wind;
    
    double w     = 0.0;
    double dwdxi  = 0.0; /* first derivative of w w.r.t xi*/
    double dwdet  = 0.0; /* first derivative of w w.r.t eta*/
    double dwdze  = 0.0; /* first derivative of w w.r.t zeta*/
    double d2wdxi = 0.0; /* second derivative of w w.r.t xi*/
    double d2wdet = 0.0; /* second derivative of w w.r.t eta*/
    double d2wdze = 0.0; /* second derivative of w w.r.t zeta*/
    double d2wdxe = 0.0; /* second derivative of w w.r.t xi-eta*/
    double d2wdez = 0.0; /* second derivative of w w.r.t eta-zeta*/
    double d2wdxz = 0.0; /* second derivative of w w.r.t xi-zeta*/
    double wgt;
    
    for(k = 0; k <= r; k++)
    {
        wind = spanW - r + k;
        
        for(j = 0; j <= q; j++)
        {
            vind = spanV - q + j;
            
            for(i = 0; i <= p; i++)
            {
                c   = uind + i + (nU+1) * ( (nV+1)*wind + vind);
                wgt = weight[c];
                
                w     += N[i]        * M[j] * P[k] * wgt;
                dwdxi += dersN[1][i] * M[j] * P[k] * wgt;
                dwdet += dersM[1][j] * N[i] * P[k] * wgt;
                dwdze += dersP[1][k] * N[i] * M[j] * wgt;
                d2wdxi += dersN[2][i] * M[j] * P[k] * wgt;
                d2wdet += dersM[2][j] * N[i] * P[k] * wgt;
                d2wdze += dersP[2][k] * N[i] * M[j] * wgt;
                d2wdxe += dersN[1][i] * dersM[1][j] * wgt;
                d2wdez += dersM[1][j] * dersP[1][k] * wgt;
                d2wdxz += dersN[1][i] * dersP[1][k] * wgt;
            }
        }
    }
    
    /* create output */
    auto result = py::array_t<double>(9 * noFuncs);
    py::buffer_info buf = result.request();
	double *ptr = static_cast<double *>(buf.ptr);
    
    uind = spanU - p;
    kk   = 0;
    
    double fac;
    double nmp, wi;
    
    for(k = 0; k <= r; k++)
    {
        wind = spanW - r + k;
        
        for(j = 0; j <= q; j++)
        {
            vind = spanV - q + j;
            
            for(i = 0; i <= p; i++)
            {
                c        = uind + i + (nU+1) * ( (nV+1)*wind + vind);
                fac      = weight[c]/(w*w);
                wi       = weight[c];
                nmp      = N[i]*M[j]*P[k];
                
                ptr[9 * kk] = nmp * fac * w;
                ptr[9 * kk + 1] = (dersN[1][i]*M[j]*P[k]*w - nmp*dwdxi) * fac;
                ptr[9 * kk + 2] = (dersM[1][j]*N[i]*P[k]*w - nmp*dwdet) * fac;
                ptr[9 * kk + 3] = (dersP[1][k]*N[i]*M[j]*w - nmp*dwdze) * fac;

                ptr[9 * kk + 4] = wi*(dersN[2][i]*M[j]*P[k]/w - 2*dersN[1][i]*M[j]*P[k]*dwdxi/w/w - N[i]*M[j]*P[k]*d2wdxi/w/w + 2*N[i]*M[j]*P[k]*dwdxi*dwdxi/w/w/w);
                ptr[9 * kk + 5] = wi*(dersM[2][j]*N[i]*P[k]/w - 2*dersM[1][j]*N[i]*P[k]*dwdet/w/w - N[i]*M[j]*P[k]*d2wdet/w/w + 2*N[i]*M[j]*P[k]*dwdet*dwdet/w/w/w);
                ptr[9 * kk + 6] = wi*(dersP[2][k]*N[i]*M[j]/w - 2*dersP[1][k]*N[i]*M[j]*dwdze/w/w - N[i]*M[j]*P[k]*d2wdze/w/w + 2*N[i]*M[j]*P[k]*dwdze*dwdze/w/w/w);

                ptr[9 * kk + 7] = wi*(dersN[1][i]*dersM[1][j]/w - dersN[1][i]*M[j]*dwdet/w/w - N[i]*dersM[1][j]*dwdxi/w/w - N[i]*M[j]*d2wdxe/w/w + 2*N[i]*M[j]*dwdxi*dwdet/w/w/w);
                ptr[9 * kk + 8] = wi*(dersM[1][j]*dersP[1][k]/w - dersM[1][j]*P[k]*dwdxi/w/w - M[j]*dersP[1][k]*dwdet/w/w - M[j]*P[k]*d2wdez/w/w + 2*M[j]*P[k]*dwdet*dwdze/w/w/w);
                ptr[9 * kk + 9] = wi*(dersN[1][i]*dersP[1][k]/w - dersP[1][k]*N[i]*dwdet/w/w - P[k]*dersN[1][i]*dwdze/w/w - N[i]*P[k]*d2wdxz/w/w + 2*N[i]*P[k]*dwdxi*dwdze/w/w/w);
                
                kk      += 1;
            }
        }
    }

    free(N);
    free(M);
    free(P);
    free2Darray(dersN, (nU+1));
    free2Darray(dersM, (nV+1));
    free2Darray(dersP, (nW+1));

    return result;
}


PYBIND11_MODULE(nurbs_basis_ders, m)
{
    m.def("nurbs_basis", &NURBSbasis, "Return the NURBS basis function to Python");;
    m.def("nurbs1d_basis_ders", &NURBS1DBasisDers, "Return the 1D NURBS basis functions and first derivatives to Python");
    m.def("nurbs1d_basis_ders_2nd_ders", &NURBS1DBasis2ndDers, "Return the 1D NURBS basis functions with first and second derivatives to Python");
    m.def("nurbs2d_basis_ders", &NURBS2DBasisDers, "Return the 2D NURBS basis functions and first derivatives to Python");
    m.def("nurbs2d_basis_ders_2nd_ders", &NURBS2DBasis2ndDers, "Return the 2D NURBS basis functions with first and second derivatives to Python");
    m.def("nurbs3d_basis_ders", &NURBS3DBasisDers, "Return the 3D NURBS basis functions and first derivatives to Python");
    m.def("nurbs3d_basis_ders_2nd_ders", &NURBS3DBasis2ndDers, "Return the 3D NURBS basis functions with first and second derivatives to Python");
}
 





