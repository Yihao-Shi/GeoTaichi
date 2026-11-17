#include "./NURBS.h"
#include "./KnotInsertion.h"


py:: array_t<double> hRefineNurbs1d(int refineCount, int p, int numPoint, py::array_t<double> PointX, py::array_t<double> PointY, py::array_t<double> PointZ, py::array_t<double> Weight, py::array_t<double> Knot)
{
  // First get the inputs //
  py::buffer_info knot_buffer = Knot.request();
	double *knot = static_cast<double *>(knot_buffer.ptr);
  py::buffer_info pointx_buffer = PointX.request();
	double *pointx = static_cast<double *>(pointx_buffer.ptr);
  py::buffer_info pointy_buffer = PointY.request();
	double *pointy = static_cast<double *>(pointy_buffer.ptr);
  py::buffer_info pointz_buffer = PointZ.request();
	double *pointz = static_cast<double *>(pointz_buffer.ptr);
  py::buffer_info weight_buffer = Weight.request();
	double *weight = static_cast<double *>(weight_buffer.ptr);

  int numKnot = knot_buffer.size;

  int numUniKnot=0, totalUniKnot=0;
  FindTotalUniqueKnotNumber(refineCount, numKnot, &numUniKnot, &totalUniKnot, knot);

  // create output //
  auto result = py::array_t<double>(4 * (numPoint + totalUniKnot) + numKnot + totalUniKnot);
  py::buffer_info buf = result.request();
	double *interp = static_cast<double *>(buf.ptr);

  double **opoint         = init2DArray(numPoint + totalUniKnot, 4);
  double *oknot           = init1DArray(numKnot + totalUniKnot);

  copyPointweightX(numPoint, 1, 1, opoint, pointx, pointy, pointz, weight);
  copy1DArray(0, numKnot, oknot, knot);
  
  int nc = numPoint, nk = numKnot;
  RefineKnotCurve(1, refineCount, p, &nc, &nk, numUniKnot, opoint, oknot);

  copyResultPointX(nc, 1, 1, opoint, interp);
  copyResultWeightX(nc, 1, 1, opoint, interp);
  copyResutlKnot(4 * nc, nk, oknot, interp);

  free2Darray(opoint, numPoint + totalUniKnot);
  free(oknot);

  return result;
}

py:: array_t<double> hRefineNurbs2d(int refineCountU, int refineCountV, int p, int q, int numPointX, int numPointY, py::array_t<double> PointX, py::array_t<double> PointY, 
                                    py::array_t<double> PointZ, py::array_t<double> Weight, py::array_t<double> KnotU, py::array_t<double> KnotV)
{
  // First get the inputs //
  py::buffer_info knotU_buffer = KnotU.request();
	double *uknot = static_cast<double *>(knotU_buffer.ptr);
  py::buffer_info knotV_buffer = KnotV.request();
	double *vknot = static_cast<double *>(knotV_buffer.ptr);
  py::buffer_info pointx_buffer = PointX.request();
	double *pointx = static_cast<double *>(pointx_buffer.ptr);
  py::buffer_info pointy_buffer = PointY.request();
	double *pointy = static_cast<double *>(pointy_buffer.ptr);
  py::buffer_info pointz_buffer = PointZ.request();
	double *pointz = static_cast<double *>(pointz_buffer.ptr);
  py::buffer_info weight_buffer = Weight.request();
	double *weight = static_cast<double *>(weight_buffer.ptr);

  int numKnotU = knotU_buffer.size;
  int numKnotV = knotV_buffer.size;

  int numUniKnotU=0, totalUniKnotU=0;
  FindTotalUniqueKnotNumber(refineCountU, numKnotU, &numUniKnotU, &totalUniKnotU, uknot);
  int numUniKnotV=0, totalUniKnotV=0;
  FindTotalUniqueKnotNumber(refineCountV, numKnotV, &numUniKnotV, &totalUniKnotV, vknot);

  // create output //
  int totalLen = 4 * (numPointX + totalUniKnotU) * (numPointY + totalUniKnotV) + numKnotU + totalUniKnotU + numKnotV + totalUniKnotV;
  auto result = py::array_t<double>(totalLen);
  py::buffer_info buf = result.request();
	double *interp = static_cast<double *>(buf.ptr);

  // u direction //
  double **opointX         = init2DArray(numPointX + totalUniKnotU, 4 * numPointY);
  double *nknotU          = init1DArray(numKnotU + totalUniKnotU);

  copyPointweightX(numPointX, numPointY, 1, opointX, pointx, pointy, pointz, weight);
  copy1DArray(0, numKnotU, nknotU, uknot);

  int ncx = numPointX, nku = numKnotU;
  RefineKnotCurve(numPointY, refineCountU, p, &ncx, &nku, numUniKnotU, opointX, nknotU);

  double *npointx         = init1DArray((numPointX + totalUniKnotU) * numPointY);
  double *npointy         = init1DArray((numPointX + totalUniKnotU) * numPointY);
  double *npointz         = init1DArray((numPointX + totalUniKnotU) * numPointY);
  double *nweight         = init1DArray((numPointX + totalUniKnotU) * numPointY);

  copyTempX((numPointX + totalUniKnotU), numPointY, 1, opointX, npointx, npointy, npointz, nweight);
  free2Darray(opointX, numPointX + totalUniKnotU);

  numPointX = ncx;
  numKnotU = nku;

  // v direction //
  double **opointY         = init2DArray((numPointY + totalUniKnotV), 4 * numPointX);
  double *nknotV          = init1DArray(numKnotV + totalUniKnotV);

  copyPointweightY(numPointX, numPointY, 1, opointY, npointx, npointy, npointz, nweight);
  copy1DArray(0, numKnotV, nknotV, vknot);

  free(npointx);
  free(npointy);
  free(npointz);
  free(nweight);
  
  int ncy = numPointY, nkv = numKnotV;
  RefineKnotCurve(numPointX, refineCountV, q, &ncy, &nkv, numUniKnotV, opointY, nknotV);

  // output //
  copyResultPointY(ncx, ncy, 1, opointY, interp);
  copyResultWeightY(ncx, ncy, 1, opointY, interp);
  copyResutlKnot(4 * ncx * ncy, nku, nknotU, interp);
  copyResutlKnot(4 * ncx * ncy + nku, nkv, nknotV, interp);

  free2Darray(opointY, numPointY + totalUniKnotV);
  free(nknotU);
  free(nknotV);

  return result;
}

py:: array_t<double> hRefineNurbs3d(int refineCountU, int refineCountV, int refineCountW, int p, int q, int r, int numPointX, int numPointY, int numPointZ, py::array_t<double> PointX, 
                                    py::array_t<double> PointY, py::array_t<double> PointZ, py::array_t<double> Weight, py::array_t<double> KnotU, py::array_t<double> KnotV, py::array_t<double> KnotW)
{
  // First get the inputs //
  py::buffer_info knotU_buffer = KnotU.request();
	double *uknot = static_cast<double *>(knotU_buffer.ptr);
  py::buffer_info knotV_buffer = KnotV.request();
	double *vknot = static_cast<double *>(knotV_buffer.ptr);
  py::buffer_info knotW_buffer = KnotW.request();
	double *wknot = static_cast<double *>(knotW_buffer.ptr);
  py::buffer_info pointx_buffer = PointX.request();
	double *pointx = static_cast<double *>(pointx_buffer.ptr);
  py::buffer_info pointy_buffer = PointY.request();
	double *pointy = static_cast<double *>(pointy_buffer.ptr);
  py::buffer_info pointz_buffer = PointZ.request();
	double *pointz = static_cast<double *>(pointz_buffer.ptr);
  py::buffer_info weight_buffer = Weight.request();
	double *weight = static_cast<double *>(weight_buffer.ptr);

  int numKnotU = knotU_buffer.size;
  int numKnotV = knotV_buffer.size;
  int numKnotW = knotW_buffer.size;

  int numUniKnotU=0, totalUniKnotU=0;
  FindTotalUniqueKnotNumber(refineCountU, numKnotU, &numUniKnotU, &totalUniKnotU, uknot);
  int numUniKnotV=0, totalUniKnotV=0;
  FindTotalUniqueKnotNumber(refineCountV, numKnotV, &numUniKnotV, &totalUniKnotV, vknot);
  int numUniKnotW=0, totalUniKnotW=0;
  FindTotalUniqueKnotNumber(refineCountW, numKnotW, &numUniKnotW, &totalUniKnotW, wknot);

  // create output //
  int totalLen = 4 * (numPointX + totalUniKnotU) * (numPointY + totalUniKnotV) * (numPointZ + totalUniKnotW) + numKnotU + totalUniKnotU + numKnotV + totalUniKnotV + numKnotW + totalUniKnotW;
  auto result = py::array_t<double>(totalLen);
  py::buffer_info buf = result.request();
	double *interp = static_cast<double *>(buf.ptr);

  // u direction //
  double **opointX         = init2DArray(numPointX + totalUniKnotU, 4 * numPointY * numPointZ);
  double *nknotU           = init1DArray(numKnotU + totalUniKnotU);

  copyPointweightX(numPointX, numPointY, numPointZ, opointX, pointx, pointy, pointz, weight);
  copy1DArray(0, numKnotU, nknotU, uknot);

  int ncx = numPointX, nku = numKnotU;
  RefineKnotCurve(numPointY * numPointZ, refineCountU, p, &ncx, &nku, numUniKnotU, opointX, nknotU);

  // update control point //
  double *npointx1         = init1DArray((numPointX + totalUniKnotU) * numPointY * numPointZ);
  double *npointy1         = init1DArray((numPointX + totalUniKnotU) * numPointY * numPointZ);
  double *npointz1         = init1DArray((numPointX + totalUniKnotU) * numPointY * numPointZ);
  double *nweight1         = init1DArray((numPointX + totalUniKnotU) * numPointY * numPointZ);

  copyTempX((numPointX + totalUniKnotU), numPointY, numPointZ, opointX, npointx1, npointy1, npointz1, nweight1);
  free2Darray(opointX, numPointX + totalUniKnotU);

  numPointX = ncx;
  numKnotU = nku;

  // v direction //
  double **opointY         = init2DArray((numPointY + totalUniKnotV), 4 * numPointX * numPointZ);
  double *nknotV          = init1DArray(numKnotV + totalUniKnotV);

  copyPointweightY(numPointX, numPointY, numPointZ, opointY, npointx1, npointy1, npointz1, nweight1);
  copy1DArray(0, numKnotV, nknotV, vknot);
  
  int ncy = numPointY, nkv = numKnotV;
  RefineKnotCurve(numPointX * numPointZ, refineCountV, q, &ncy, &nkv, numUniKnotV, opointY, nknotV);

  // update control point //
  free(npointx1);
  free(npointy1);
  free(npointz1);
  free(nweight1);

  double *npointx2         = init1DArray((numPointY + totalUniKnotV) * numPointX * numPointZ);
  double *npointy2         = init1DArray((numPointY + totalUniKnotV) * numPointX * numPointZ);
  double *npointz2         = init1DArray((numPointY + totalUniKnotV) * numPointX * numPointZ);
  double *nweight2         = init1DArray((numPointY + totalUniKnotV) * numPointX * numPointZ);

  copyTempY(numPointX, (numPointY + totalUniKnotV), numPointZ, opointY, npointx2, npointy2, npointz2, nweight2);
  free2Darray(opointY, numPointY + totalUniKnotV);

  numPointY = ncy;
  numKnotV = nkv;

  // w direction //
  double **opointZ         = init2DArray((numPointZ + totalUniKnotW), 4 * numPointX * numPointY);
  double *nknotW          = init1DArray(numKnotW + totalUniKnotW);

  copyPointweightZ(numPointX, numPointY, numPointZ, opointZ, npointx2, npointy2, npointz2, nweight2);
  copy1DArray(0, numKnotW, nknotW, wknot);

  free(npointx2);
  free(npointy2);
  free(npointz2);
  free(nweight2);
  
  int ncz = numPointZ, nkw = numKnotW;
  RefineKnotCurve(numPointX * numPointY, refineCountW, r, &ncz, &nkw, numUniKnotW, opointZ, nknotW);

  // output //
  copyResultPointZ(ncx, ncy, ncz, opointZ, interp);
  copyResultWeightZ(ncx, ncy, ncz, opointZ, interp);
  copyResutlKnot(4 * ncx * ncy * ncz, nku, nknotU, interp);
  copyResutlKnot(4 * ncx * ncy * ncz + nku, nkv, nknotV, interp);
  copyResutlKnot(4 * ncx * ncy * ncz + nku + nkv, nkw, nknotW, interp);

  free2Darray(opointZ, numPointZ + totalUniKnotW);
  free(nknotU);
  free(nknotV);
  free(nknotW);

  return result;
}


PYBIND11_MODULE(hrefine_nurbs, m)
{
    m.def("hrefine_nurbs1d", &hRefineNurbs1d, "Insert knots into Non uniform B-Spline");
    m.def("hrefine_nurbs2d", &hRefineNurbs2d, "Insert knots into Non uniform B-Spline");
    m.def("hrefine_nurbs3d", &hRefineNurbs3d, "Insert knots into Non uniform B-Spline");
}