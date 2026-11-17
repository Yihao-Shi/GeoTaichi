class LinearOperator(object):
    def __init__(self, matvec_kernel):
        self.matvec = matvec_kernel
