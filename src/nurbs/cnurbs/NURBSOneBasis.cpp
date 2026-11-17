#include "./NURBS.h"


py:: array_t<double> NURBSOneBasis(int i, int p, double xi, py::array_t<double> in_knot, py::array_t<double> in_weights)
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


PYBIND11_MODULE(nurbs_one_basis, m)
{
    m.def("nurbs_one_basis", &NURBSOneBasis, "Return the NURBS basis function to Python");;
}
 
 

