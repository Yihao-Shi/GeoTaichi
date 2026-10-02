import taichi as ti

from src.linear_solver.ConjugateGradientSolverKernel import (
    M_init as _M_init,
    compute_Ad as _compute_Ad,
    compute_rMr as _compute_rMr,
    dot_product as _dot_product,
    r_d_init as _r_d_init,
    rmax as _rmax,
    update_d as _update_d,
    update_r as _update_r,
    update_x as _update_x,
)


@ti.data_oriented
class ConjugateGradientSolver_rowMajor:

    def __init__(
        self,
        spm: ti.template(),
        sparseIJ: ti.template(),  # this is used to define the sparse matrix
        b: ti.template(),
        eps=1.0e-3,  # the allowable relative error of residual
    ):
        self.A = spm  # sparse matrix, also called stiffness matrix or coefficient matrix
        self.ij = sparseIJ  # for each row, record the colume index of sparse matrix (each row, index 0 stores the number of effective indexes)
        self.b = b  # the right hand side (rhs) of the linear system

        self.x = ti.field(float, b.shape[0])  # the solution x
        self.r = ti.field(float, b.shape[0])  # the residual
        self.d = ti.field(float, b.shape[0])  # the direction of change of x
        self.M = ti.field(float, b.shape[0])
        self.M_init()  # the inverse of precondition diagonal matrix, M^(-1) actually

        self.Ad = ti.field(float, b.shape[0])  # A multiply d
        self.eps = eps

    def re_init(
        self,
    ):
        """re_initialize if this CG class is reused repeatedly"""
        self.x.fill(0.0)
        self.r.fill(0.0)
        self.d.fill(0.0)
        self.M.fill(0.0)
        self.M_init()
        self.Ad.fill(0.0)

    def solve(
        self,
    ):
        self.r_d_init()
        r0 = self.rmax()  # the inital residual scale
        print("\033[32;1m the initial residual scale is {} \033[0m".format(r0))

        for i in range(self.b.shape[0]):  # CG will converge within at most b.shape[0] loops
            self.compute_Ad()
            rMr = self.compute_rMr()
            alpha = rMr / self.dot_product(self.d, self.Ad)
            self.update_x(alpha)
            self.update_r(alpha)
            beta = self.compute_rMr() / rMr
            self.update_d(beta)

            rmax = self.rmax()  # the infinite norm of residual, shold be modified latter to the reduce max

            if rmax < self.eps * r0:  # converge?
                break

    def M_init(self):
        _M_init(self.M, self.A, self.ij)

    def compute_Ad(self):
        _compute_Ad(self.A, self.ij, self.d, self.Ad)

    def r_d_init(self):
        _r_d_init(self.b, self.M, self.r, self.d)

    def rmax(self):
        return _rmax(self.r)

    def compute_rMr(self):
        return _compute_rMr(self.r, self.M)

    def update_x(self, alpha):
        _update_x(self.x, self.d, alpha)

    def update_r(self, alpha):
        _update_r(self.r, self.Ad, alpha)

    def update_d(self, beta):
        _update_d(self.d, self.M, self.r, beta)

    @staticmethod
    def dot_product(y, z):
        return _dot_product(y, z)
