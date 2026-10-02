#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

void     FindTotalUniqueKnotNumber(int refineCount, int numKnot, int *numUniKnot, int *totalUniKnot, double *knot);
void     FindMultiplicity(int size, double knot_val, double *knot, int *mult);
void     RefineKnotCurve(int colNum, int refineCount, int p, int* nc, int* nk, int numUniKnot, double **opoint, double *oknot);
void     bspkntins(int colNum, int p, int nc, int nu, double **opoint, double **npoint, double* oknot, double* nknot, double* knotVector);
void     copyResultPointX(int xdim, int ydim, int zdim, double **opoint, double* result);
void     copyResultPointY(int xdim, int ydim, int zdim, double **opoint, double* result);
void     copyResultPointZ(int xdim, int ydim, int zdim, double **opoint, double* result);
void     copyResultWeightX(int xdim, int ydim, int zdim, double **opoint, double* result);
void     copyResultWeightY(int xdim, int ydim, int zdim, double **opoint, double* result);
void     copyResultWeightZ(int xdim, int ydim, int zdim, double **opoint, double* result);
void     copyResutlKnot(int nc, int nk, double *oknot, double* result);
void     copyPointweightX(int xdim, int ydim, int zdim, double **points, double* pointx, double* pointy, double* pointz, double* weight);
void     copyPointweightY(int xdim, int ydim, int zdim, double **points, double* pointx, double* pointy, double* pointz, double* weight);
void     copyPointweightZ(int xdim, int ydim, int zdim, double **points, double* pointx, double* pointy, double* pointz, double* weight);
void     copyTempX(int xdim, int ydim, int zdim, double**points, double* pointx, double* pointy, double* pointz, double* weight);
void     copyTempY(int xdim, int ydim, int zdim, double**points, double* pointx, double* pointy, double* pointz, double* weight);


void copyTempX(int xdim, int ydim, int zdim, double**points, double* pointx, double* pointy, double* pointz, double* weight)
{
	int idx, idy, idz, row, column;
    for(int i = 0; i < xdim * ydim * zdim; i++)
	{
		idx = (i % (xdim * ydim)) % xdim;
        idy = (i % (xdim * ydim)) / xdim;
        idz = i / (xdim * ydim);
        row = idx;
        column = idy + idz * ydim;

        pointx[i] = points[row][4 * column + 0] / points[row][4 * column + 3];
		pointy[i] = points[row][4 * column + 1] / points[row][4 * column + 3];
		pointz[i] = points[row][4 * column + 2] / points[row][4 * column + 3];
		weight[i] = points[row][4 * column + 3];
	}
}

void copyTempY(int xdim, int ydim, int zdim, double**points, double* pointx, double* pointy, double* pointz, double* weight)
{
	int idx, idy, idz, row, column;
    for(int i = 0; i < xdim * ydim * zdim; i++)
	{
		idx = (i % (xdim * ydim)) % xdim;
        idy = (i % (xdim * ydim)) / xdim;
        idz = i / (xdim * ydim);
        row = idy;
        column = idx + idz * xdim;

        pointx[i] = points[row][4 * column + 0] / points[row][4 * column + 3];
		pointy[i] = points[row][4 * column + 1] / points[row][4 * column + 3];
		pointz[i] = points[row][4 * column + 2] / points[row][4 * column + 3];
		weight[i] = points[row][4 * column + 3];
	}
}

void copyPointweightX(int xdim, int ydim, int zdim, double**points, double* pointx, double* pointy, double* pointz, double* weight)
{
	int idx, idy, idz, row, column;
    for(int i = 0; i < xdim * ydim * zdim; i++)
	{
		idx = (i % (xdim * ydim)) % xdim;
        idy = (i % (xdim * ydim)) / xdim;
        idz = i / (xdim * ydim);
        row = idx;
        column = idy + idz * ydim;

        points[row][4 * column + 0] = pointx[i] * weight[i];
		points[row][4 * column + 1] = pointy[i] * weight[i];
		points[row][4 * column + 2] = pointz[i] * weight[i];
		points[row][4 * column + 3] = weight[i];
	}
}

void copyPointweightY(int xdim, int ydim, int zdim, double**points, double* pointx, double* pointy, double* pointz, double* weight)
{
	int idx, idy, idz, row, column;
    for(int i = 0; i < xdim * ydim * zdim; i++)
	{
		idx = (i % (xdim * ydim)) % xdim;
        idy = (i % (xdim * ydim)) / xdim;
        idz = i / (xdim * ydim);
        row = idy;
        column = idx + idz * xdim;

        points[row][4 * column + 0] = pointx[i] * weight[i];
		points[row][4 * column + 1] = pointy[i] * weight[i];
		points[row][4 * column + 2] = pointz[i] * weight[i];
		points[row][4 * column + 3] = weight[i];
	}
}

void copyPointweightZ(int xdim, int ydim, int zdim, double**points, double* pointx, double* pointy, double* pointz, double* weight)
{
	int idx, idy, idz, row, column;
    for(int i = 0; i < xdim * ydim * zdim; i++)
	{
		idx = (i % (xdim * ydim)) % xdim;
        idy = (i % (xdim * ydim)) / xdim;
        idz = i / (xdim * ydim);
        row = idz;
        column = idx + idy * xdim;

        points[row][4 * column + 0] = pointx[i] * weight[i];
		points[row][4 * column + 1] = pointy[i] * weight[i];
		points[row][4 * column + 2] = pointz[i] * weight[i];
		points[row][4 * column + 3] = weight[i];
	}
}

void copyResultPointX(int xdim, int ydim, int zdim, double **opoint, double* result)
{
    int number = xdim * ydim * zdim;
    int idx, idy, idz, row, column;
    for(int i = 0; i < number; i++)  
    {   
        idx = (i % (xdim * ydim)) % xdim;
        idy = (i % (xdim * ydim)) / xdim;
        idz = i / (xdim * ydim);
        row = idx;
        column = idy + idz * ydim;

        result[3 * i] = opoint[row][4 * column + 0] / opoint[row][4 * column + 3];
        result[3 * i + 1] = opoint[row][4 * column + 1] / opoint[row][4 * column + 3];
        result[3 * i + 2] = opoint[row][4 * column + 2] / opoint[row][4 * column + 3];
    }
}

void copyResultPointY(int xdim, int ydim, int zdim, double **opoint, double* result)
{
    int number = xdim * ydim * zdim;
    int idx, idy, idz, row, column;
    for(int i = 0; i < number; i++)  
    {   
        idx = (i % (xdim * ydim)) % xdim;
        idy = (i % (xdim * ydim)) / xdim;
        idz = i / (xdim * ydim);
        row = idy;
        column = idx + idz * xdim;

        result[3 * i] = opoint[row][4 * column + 0] / opoint[row][4 * column + 3];
        result[3 * i + 1] = opoint[row][4 * column + 1] / opoint[row][4 * column + 3];
        result[3 * i + 2] = opoint[row][4 * column + 2] / opoint[row][4 * column + 3];
    }
}

void copyResultPointZ(int xdim, int ydim, int zdim, double **opoint, double* result)
{
    int number = xdim * ydim * zdim;
    int idx, idy, idz, row, column;
    for(int i = 0; i < number; i++)  
    {   
        idx = (i % (xdim * ydim)) % xdim;
        idy = (i % (xdim * ydim)) / xdim;
        idz = i / (xdim * ydim);
        row = idz;
        column = idx + idy * xdim;

        result[3 * i] = opoint[row][4 * column + 0] / opoint[row][4 * column + 3];
        result[3 * i + 1] = opoint[row][4 * column + 1] / opoint[row][4 * column + 3];
        result[3 * i + 2] = opoint[row][4 * column + 2] / opoint[row][4 * column + 3];
    }
}

void copyResultWeightX(int xdim, int ydim, int zdim, double **opoint, double* result)
{
    int number = xdim * ydim * zdim;
    int idx, idy, idz, row, column;
    for(int i = 0; i < number; i++)  
    {
        idx = (i % (xdim * ydim)) % xdim;
        idy = (i % (xdim * ydim)) / xdim;
        idz = i / (xdim * ydim);
        row = idx;
        column = idy + idz * ydim;

        result[3 * number + i] = opoint[row][4 * column + 3];
    }
}

void copyResultWeightY(int xdim, int ydim, int zdim, double **opoint, double* result)
{
    int number = xdim * ydim * zdim;
    int idx, idy, idz, row, column;
    for(int i = 0; i < number; i++)  
    {
        idx = (i % (xdim * ydim)) % xdim;
        idy = (i % (xdim * ydim)) / xdim;
        idz = i / (xdim * ydim);
        row = idy;
        column = idx + idz * xdim;

        result[3 * number + i] = opoint[row][4 * column + 3];
    }
}

void copyResultWeightZ(int xdim, int ydim, int zdim, double **opoint, double* result)
{
    int number = xdim * ydim * zdim;
    int idx, idy, idz, row, column;
    for(int i = 0; i < number; i++)  
    {   
        idx = (i % (xdim * ydim)) % xdim;
        idy = (i % (xdim * ydim)) / xdim;
        idz = i / (xdim * ydim);
        row = idz;
        column = idx + idy * xdim;
        
        result[3 * number + i] = opoint[row][4 * column + 3];
    }
}

void copyResutlKnot(int nc, int nk, double *oknot, double* result)
{
    for(int i = 0; i < nk; i++)
    {
        result[nc + i] = oknot[i];
    }
}

void FindTotalUniqueKnotNumber(int refineCount, int numKnot, int *numUniKnot, int *totalUniKnot, double *knot)
{
    double *knot_copy                  = init1DArray(numKnot);
    for(int i = 0; i < numKnot; i++)         knot_copy[i] = knot[i];
    *numUniKnot                     = unique(knot_copy, knot_copy + numKnot) - knot_copy;
    *totalUniKnot                   = *numUniKnot;
    for(int i = 0; i < refineCount; i++)     *totalUniKnot = 2 * (*totalUniKnot) - 1;
    *totalUniKnot                       -= *numUniKnot;
    free(knot_copy);
}

void FindMultiplicity(int size, double knot_val, double *knot, int *mult)
{
    for(int i = 0; i < size; i++)
    {
        if(knot_val == knot[i])  mult++;
    }
}

void RefineKnotCurve(int colNum, int refineCount, int p, int* nc, int* nk, int numUniKnot, double **opoint, double *oknot)
{
    int re, i, nu, nuni;
    for(re = 0; re < refineCount; re++)
    {
        double *knot_copy                  = init1DArray(*nk);
        copy1DArray(0, *nk, knot_copy, oknot); 

        nuni                               = unique(knot_copy, knot_copy + *nk) - knot_copy;
        nu                                 = nuni-1;
        double *knotVector                 = init1DArray(nu);
        for(i = 0; i < nuni - 1; i++)
        {
            knotVector[i] = knot_copy[i] + 0.5 * (knot_copy[i + 1] - knot_copy[i]);
        }
        free(knot_copy);

        double **npoint         = init2DArray(*nc + nu, 4 * colNum);
        double *nknot           = init1DArray(*nk + nu);

        bspkntins(colNum, p, *nc, nu, opoint, npoint, oknot, nknot, knotVector);
        copy2DArray(0, *nc + nu, 0, 4 * colNum, opoint, npoint);
        copy1DArray(0, *nk + nu, oknot, nknot);

        free2Darray(npoint, *nc + nu);
        free(nknot);
        free(knotVector);

        *nk     += nu;
        *nc     += nu;
    }
}


void bspkntins(int colNum, int p, int nc, int nu, double **opoint, double **npoint, double* oknot, double* nknot, double* knotVector)
{
    int n = nc - 1;
    int r = nu - 1;
    int m = n + p + 1;
    
    int l, i, j, s, ind;
    double alfa;
    
    int a = (p + n + 1) / 2; 
    int b = (p + n + 1) / 2; 
    FindSpan(n, p, knotVector[0], oknot, &a);
    FindSpan(n, p, knotVector[r], oknot, &b);
    ++b;
    
    copy2DArray(0, a - p + 1, 0, 4 * colNum, npoint, opoint);
    for (j = b - 1; j <= n; j++)
    {
        copy1DArray(0, 4 * colNum, *(npoint + j + r + 1), *(opoint + j));
    }
    copy1DArray(0, a + 1, nknot, oknot);
    for (j = b + p; j <= m; j++)    nknot[j + r + 1] = oknot[j];

    i = b + p - 1;
    s = b + p + r;
    for (j = r; j >= 0; j--) 
    {
        while (knotVector[j] <= oknot[i] && i > a) 
        {
            copy1DArray(0, 4 * colNum, *(npoint+s-p-1), *(opoint+i-p-1));
            nknot[s] = oknot[i];
            --s;
            --i;
        }

        copy1DArray(0, 4 * colNum, *(npoint+s-p-1), *(npoint+s-p));
        for (l = 1; l <= p; l++)  
        {
            ind = s - p + l;
            alfa = nknot[s + l] - knotVector[j];
            if (fabs(alfa) == 0.0)
            {
                copy1DArray(0, 4 * colNum, *(npoint+ind-1), *(npoint+ind));
            }
            else  
            {
                alfa /= (nknot[s + l] - oknot[i - p + l]);
                for(int q = 0; q < 4 * colNum; q++)
                {
                    npoint[ind - 1][q] = alfa * npoint[ind - 1][q] + (1.0 - alfa) * npoint[ind][q];
                }
            }
        }
        nknot[s] = knotVector[j];
        --s;
    }
}