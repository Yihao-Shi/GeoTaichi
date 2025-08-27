class LinearOperator(object):
    def __init__(self, matvec_kernel):
        self._matvec = matvec_kernel
        self.matvec = self.matvec_go

    def matvec_go(self, x, Ax):
        self._matvec(x, Ax)