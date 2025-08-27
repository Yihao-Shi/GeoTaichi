import taichi as ti


@ti.data_oriented
class ConjugateGradientSolver_rowMajor:

    def __init__(self, spm: ti.template(), sparseIJ: ti.template(),  # this is used to define the sparse matrix
                 b: ti.template(),
                 eps=1.e-3,  # the allowable relative error of residual 
                 ):
        self.A = spm  # sparse matrix, also called stiffness matrix or coefficient matrix
        self.ij = sparseIJ  # for each row, record the colume index of sparse matrix (each row, index 0 stores the number of effective indexes)
        self.b = b  # the right hand side (rhs) of the linear system

        self.x = ti.field(float, b.shape[0])  # the solution x
        self.r = ti.field(float, b.shape[0])  # the residual
        self.d = ti.field(float, b.shape[0])  # the direction of change of x
        self.M = ti.field(float, b.shape[0]); self.M_init()  # the inverse of precondition diagonal matrix, M^(-1) actually

        self.Ad = ti.field(float, b.shape[0])  # A multiply d
        self.eps = eps
    

    def re_init(self, ): 
        """re_initialize if this CG class is reused repeatedly"""
        self.x.fill(0.)
        self.r.fill(0.)
        self.d.fill(0.)
        self.M.fill(0.); self.M_init()
        self.Ad.fill(0.)

    def solve(self, ):
        r_d_init()
        r0 = rmax()  # the inital residual scale
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

@ti.func
def A_get(self, i, j):
    target_j = 0
    for j0 in range(self.ij[i][0]):
        if self.ij[i][j0 + 1] == j:
            target_j = j0
    return self.A[i][target_j]


@ti.kernel
def M_init(self, ):  # initialize the precondition diagonal matrix
    for i in self.M:
        self.M[i] = 1. / self.A_get(i, i)


@ti.kernel
def compute_Ad(self, ):  # compute A multiple d
    for i in self.A:
        self.Ad[i] = 0.
        for j0 in range(self.ij[i][0]):
            self.Ad[i] = self.Ad[i] + self.A[i][j0] * self.d[self.ij[i][j0 + 1]]


@ti.kernel
def r_d_init(self, ):  # initial residual r and direction d
    for i in self.r:  # r0 = b - Ax0 = b
        self.r[i] = self.b[i]
    for i in self.d:
        self.d[i] = self.M[i] * self.r[i]  # d0 = M^(-1) * r


@ti.kernel
def rmax(self, ) -> float:  # max of abs(r), modified latter by reduce_max
    rm = 0.
    for i in self.r:
        ti.atomic_max(rm, ti.abs(self.r[i]))
    return rm


@ti.kernel
def compute_rMr(self, ) -> float:  # r * M^(-1) * r
    rMr = 0.
    for i in self.r:
        rMr += self.r[i] * self.M[i] * self.r[i]
    return rMr


@ti.kernel 
def update_x(self, alpha: float):
    for j in self.x:
        self.x[j] = self.x[j] + alpha * self.d[j]


@ti.kernel
def update_r(self, alpha: float):
    for j in self.r:
        self.r[j] = self.r[j] - alpha * self.Ad[j]


@ti.kernel 
def update_d(self, beta: float):
    for j in self.d:
        self.d[j] = self.M[j] * self.r[j] + beta * self.d[j]


@ti.kernel
def dot_product(y: ti.template(), z: ti.template()) -> float:
    res = 0.
    for i in y:
        res += y[i] * z[i]
    return res


    