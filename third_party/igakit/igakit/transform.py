import numpy as np

__all__ = ['transform']

class transform(object):

    def __init__(self, matrix=None):
        if matrix is None:
            matrix = np.identity(4, dtype='d')
        elif isinstance(matrix, transform):
            matrix = matrix.matrix.copy()
        else:
            matrix = np.array(matrix, dtype='d')
            assert matrix.shape == (4, 4)
        self.matrix = matrix

    def __call__(self, xyzw):
        return np.dot(xyzw, self.matrix.T)

    def __array__(self, dtype='d'):
        return self.matrix.astype(dtype)

    def copy(self):
        trans = transform.__new__(type(self))
        trans.matrix = self.matrix.copy()
        return trans

    def clone(self):
        trans = transform.__new__(type(self))
        trans.matrix = self.matrix
        return trans

    def inverse(self):
        return self.clone().invert()

    def invert(self):
        self.matrix = np.linalg.inv(self.matrix).copy()
        return self

    def compose(self, matrix):
        if isinstance(matrix, transform):
            M = matrix.matrix
        else:
            matrix = np.asarray(matrix, dtype='d')
            if matrix.shape == (3, 3):
                M = np.zeros((4, 4), dtype='d')
                M[:3,:3] = matrix
                M[ 3, 3] = 1
            else:
                assert matrix.shape == (4, 4)
                M = matrix
        self.matrix = np.dot(M, self.matrix)
        return self

    def translate(self, displ, axis=None):
        displ = np.asarray(displ, dtype='d')
        assert displ.ndim in (0, 1)
        if displ.ndim > 0 and axis is None:
            assert displ.size <= 3
            t = displ
        else:
            t = np.zeros(3, dtype='d')
            if axis is None:
                t[0] = displ
            else:
                t[axis] = displ
        T = np.identity(4, dtype='d')
        T[:t.size, 3] = t
        self.compose(T)
        return self

    move = translate

    def scale(self, scale, axis=None):
        scale = np.asarray(scale, dtype='d')
        assert scale.ndim in (0, 1)
        if scale.ndim > 0 and axis is None:
            assert scale.size <= 3
            s = scale
        else:
            s = np.ones(3, dtype='d')
            if axis is None:
                s[:] = scale
            else:
                s[axis] = scale
        S = np.identity(3, dtype='d')
        i = np.arange(s.size)
        S[i,i] = s
        self.compose(S)
        return self

    def rotate(self, angle, axis=2):
        angle = float(angle)
        if angle == 0.0: return self
        axis = np.asarray(axis)
        assert axis.ndim in (0, 1)
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)
        R = np.identity(3, dtype='d')
        if axis.ndim > 0:
            assert 1 <= axis.size <= 3
            u = np.zeros(3, dtype='d')
            u[0:axis.size] = axis
            u_norm = np.linalg.norm(u)
            assert u_norm > 0
            u /= u_norm
            # R = cos(a) I + sin(a) U_x + (1-cos(a)) UU
            u_upper = np.array([-u[2], +u[1], -u[0]])
            u_lower = np.array([+u[2], -u[1], +u[0]])
            u_outer = np.outer(u, u)
            R.flat[[0,4,8]] = cos_a
            R.flat[[1,2,5]] = sin_a*u_upper
            R.flat[[3,6,7]] = sin_a*u_lower
            R += (1-cos_a)*u_outer
        else:
            axis = (0,1,2)[int(axis)]
            if axis == 0:
                R[1, [1,2]] = [ cos_a, -sin_a]
                R[2, [1,2]] = [ sin_a,  cos_a]
            elif axis == 1:
                R[0, [0,2]] = [ cos_a,  sin_a]
                R[2, [0,2]] = [-sin_a,  cos_a]
            elif axis == 2:
                R[0, [0,1]] = [ cos_a, -sin_a]
                R[1, [0,1]] = [ sin_a,  cos_a]
        self.compose(R)
        return self
