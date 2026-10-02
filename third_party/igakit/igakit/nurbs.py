import numpy as np
from igakit.transform import transform
from igakit import igalib as _igalib

_bsp = _igalib.bsp

try:
    _np_unique = np.unique
except AttributeError:
    try:
        _np_unique = np.lib.arraysetops.unique
    except AttributeError:
        _np_unique = np.unique1d


__all__ = ['NURBS', 'transform']

class NURBS(object):
    """
    An abstract NURBS object class.

    This NURBS class allows for the definition of B-spline or NURBS
    curves/surfaces/volumes by specifying a control point array, a
    sequence of knot vectors and optional rational weights.

    Parameters
    ----------
    knots : sequence of array_like
        Knot vectors. The number of knot vectors will define what kind
        of NURBS object is created (1=curve, 2=surface, 3=volume).
    control : array_like or 2-tuple of array_like
        Control points and optional rational weights.
    fields : array_like, optional
       Additional fields.
    weights : array_like, optional
       Rational weights. If weights are omitted, the object will be
       non-rational (B-spline).

    Attributes
    ----------
    dim : int
        Parametric dimension of the NURBS object {1,2,3}
    shape : tuple of ints
        Number of control points in each parametric dimension.
    degree : tuple of ints
        Polynomial degrees in each parametric dimension.
    knots : tuple of numpy.ndarray
        Knot vectors in each parametric dimension.
    array: numpy.ndarray
        Container for control points, weights, and fields.
    control : numpy.ndarray
        Control points in homogeneous 4D space (includes rational weights).
    weigths : numpy.ndarray
        Rational weigths.
    points : numpy.ndarray
        Control points projected into Cartesian 3D space.
    fields : numpy.ndarray or None
        Additional fields.

    Examples
    --------

    Create a quarter circle NURBS curve with 2D control points and
    rational weigths and check error:

    >>> C = [[0, 1], [1, 1], [1, 0]] # 3x2 grid of 2D control points
    >>> w = [1, np.sqrt(2)/2, 1]     # rational weigths
    >>> U = [0,0,0, 1,1,1]           # knot vector
    >>> crv = NURBS([U], C, weights=w)
    >>> u = np.linspace(0,1,1000)
    >>> xyz = crv(u)
    >>> x, y, z = xyz.T
    >>> r = np.sqrt(x**2+y**2)
    >>> np.allclose(r, 1, rtol=0, atol=1e-15)
    True
    >>> np.allclose(z, 0, rtol=0, atol=1e-15)
    True

    Create a quarter circle NURBS curve with homogeneous 4D control
    points and check error:

    >>> wgt = np.sqrt(2)/2
    >>> Cw = np.zeros((3,4))
    >>> Cw[0,:] = [0.0, 1.0, 0.0, 1.0]
    >>> Cw[1,:] = [wgt, wgt, 0.0, wgt]
    >>> Cw[2,:] = [1.0, 0.0, 0.0, 1.0]
    >>> crv = NURBS([U], Cw)
    >>> u = np.linspace(0,1,1000)
    >>> xyz = crv(u)
    >>> x, y, z = xyz.T
    >>> r = np.sqrt(x**2+y**2)
    >>> np.allclose(r, 1, rtol=0, atol=1e-15)
    True
    >>> np.allclose(z, 0, rtol=0, atol=1e-15)
    True

    Create a random B-spline curve:

    >>> C = np.random.rand(3,3) # 3D control points
    >>> U = [0,0,0, 1,1,1]      # knot vector
    >>> crv = NURBS([U], C)
    >>> crv.dim
    1
    >>> crv.shape
    (3,)
    >>> crv.degree
    (2,)
    >>> np.allclose(crv.knots[0], U, rtol=0, atol=1e-15)
    True
    >>> np.allclose(crv.points,   C, rtol=0, atol=1e-15)
    True
    >>> np.allclose(crv.weights,  1, rtol=0, atol=1e-15)
    True

    Create a random B-spline surface:

    >>> C = np.random.rand(3,2,3) # 3x2 grid of 3D control points
    >>> U = [0,0,0, 1,1,1]        # knot vector
    >>> V = [0,0, 1,1]            # knot vector
    >>> srf = NURBS([U,V], C)
    >>> srf.dim
    2
    >>> srf.shape
    (3, 2)
    >>> srf.degree
    (2, 1)
    >>> np.allclose(srf.knots[0], U, rtol=0, atol=1e-15)
    True
    >>> np.allclose(srf.knots[1], V, rtol=0, atol=1e-15)
    True
    >>> np.allclose(srf.points,   C, rtol=0, atol=1e-15)
    True
    >>> np.allclose(srf.weights,  1, rtol=0, atol=1e-15)
    True

    Create a random B-spline volume:

    >>> C = np.random.rand(3,2,7,3) # 3x2x7 grid of 3D control points
    >>> U = [0,0,0, 1,1,1]          # knot vector
    >>> V = [0,0, 1,1]              # knot vector
    >>> W = [0]*4+[0.25, 0.5, 0.5]+[1]*4
    >>> vol = NURBS([U,V,W], C)
    >>> vol.dim
    3
    >>> vol.shape
    (3, 2, 7)
    >>> vol.degree
    (2, 1, 3)
    >>> np.allclose(vol.knots[0], U, rtol=0, atol=1e-15)
    True
    >>> np.allclose(vol.knots[1], V, rtol=0, atol=1e-15)
    True
    >>> np.allclose(vol.knots[2], W, rtol=0, atol=1e-15)
    True
    >>> np.allclose(vol.points,   C, rtol=0, atol=1e-15)
    True
    >>> np.allclose(vol.weights,  1, rtol=0, atol=1e-15)
    True

    """

    def __init__(self, knots, control=None, fields=None, weights=None):
        """
        Create a NURBS object.
        """
        #
        knots = [np.asarray(k, dtype='d') for k in knots]
        assert len(knots) >= 1
        assert len(knots) <= 3
        for k in knots:
            assert k.ndim == 1
            assert k.size >= 4
        #
        if (isinstance(control, tuple) and len(control) == 2):
            assert weights is None
            control, weights = control
        if control is not None:
            control = np.asarray(control, dtype='d')
        if weights is not None:
            weights = np.asarray(weights, dtype='d')
        if fields is not None:
            fields = np.asarray(fields, dtype='d')
        #
        if control is None:
            assert weights is None
            dim = len(knots)
            degs = [1] * dim
            for i, U in enumerate(knots):
                p = 1
                while U[p] == U[p+1]:
                    p += 1
                degs[i] = p
            Greville = _bsp.Greville
            axes = [Greville(p, U) for p, U in zip(degs, knots)]
            shape = [len(x) for x in axes]
            Cw = np.zeros(shape+[4], dtype='d')
            for i, x in enumerate(axes):
                index = [np.newaxis] * dim
                index[i] = slice(None)
                Cw[...,i] = x[tuple(index)]
            Cw[...,3] = 1
        else:
            assert control.ndim >= 2
            sdim = control.shape[-1]
            assert sdim >= 2
            assert sdim <= 4
            if sdim < 4:
                shape = control.shape[:-1]
                Cw = np.zeros(shape + (4,), dtype='d')
                Cw[...,0:sdim] = control
                if weights is not None:
                    assert weights.shape == shape
                    Cw[...,:sdim] *= weights[...,np.newaxis]
                    Cw[...,3] = weights
                else:
                    Cw[...,3] = 1
            else:
                assert weights is None
                Cw = control
        #
        if fields is not None:
            shape = Cw.shape[:-1]
            if fields.shape == shape:
                D = fields[...,np.newaxis]
            else:
                assert fields.ndim-1 == len(shape)
                assert fields.shape[:-1] == shape
                D = fields
            w = Cw[...,3,np.newaxis]
            array = np.concatenate([Cw, D*w], axis=-1)
        else:
            array = Cw
        #
        assert array.ndim - 1 == len(knots)
        N = [n-1 for n in array.shape[:-1]]
        M = [len(k)-1 for k in knots]
        for n, m in zip(N, M):
            degree = m - n - 1
            assert degree >= 1
        #
        self._array = np.ascontiguousarray(array)
        self._knots = tuple(knots)

    @property
    def dim(self):
        """
        Parametric dimension of the NURBS object {1,2,3}.
        """
        return len(self.knots)

    @property
    def shape(self):
        """
        Shape of the control point net. Also the number of
        basis functions in each parametric direction.
        """
        return self._array.shape[:-1]

    @property
    def degree(self):
        """
        Polynomial degrees for each parametric direction.
        """
        N = [n-1 for n in self.shape]
        M = [len(k)-1 for k in self.knots]
        return tuple(m-n-1 for (m, n) in zip(M, N))

    @property
    def knots(self):
        """
        Knot vectors for each parametric dimension.
        """
        return self._knots

    @property
    def array(self):
        """
        Container for control points, weights, and fields.
        """
        return self._array

    @property
    def control(self):
        """
        Control point grid in homogeneous 4D space.
        """
        return self.array[...,:4]

    @property
    def weights(self):
        """
        Rational weight grid.
        """
        return self.array[...,3]

    @property
    def points(self):
        """
        Control point grid projected into Cartesian 3D space.
        """
        C = self.array[...,:3]
        w = self.weights[...,np.newaxis]
        return C / w

    @property
    def fields(self):
        """
        Control variables.
        """
        if self.array.shape[-1] <= 4: return None
        D = self.array[...,4:]
        w = self.weights[...,np.newaxis]
        return D / w

    #

    def spans(self, axis=None):
        """
        Non-empty span indices.
        """
        axes = range(self.dim)
        if axis is None:
            return [self.spans(i) for i in axes]
        i = axes[axis]
        p = self.degree[i]
        U = self.knots[i]
        SpanIndex = _bsp.SpanIndex
        return SpanIndex(p, U)

    def breaks(self, axis=None, mults=False):
        """
        Breaks (unique knot values) and multiplicities.
        """
        axes = range(self.dim)
        if axis is None:
            return [self.breaks(i, mults) for i in axes]
        i = axes[axis]
        p = self.degree[i]
        U = self.knots[i]
        if not mults:
            return _np_unique(U[p:-p])
        u, idx = _np_unique(U[p:-p], return_inverse=True)
        s = np.bincount(idx).astype('i')
        FindMult = _bsp.FindMult
        m = len(U)-1; n = m-p-1;
        for k, j in ((p, 0), (n, -1)):
            mult = FindMult(p, U, u[j], k)
            s[j] = min(mult, p)
        return u, s

    def greville(self, axis=None):
        """
        Greville abscissae.
        """
        axes = range(self.dim)
        if axis is None:
            return [self.greville(i) for i in axes]
        i = axes[axis]
        p = self.degree[i]
        U = self.knots[i]
        Greville = _bsp.Greville
        u = Greville(p, U)
        return u

    def lagrange(self, axis=None):
        """
        Lagrange abscissae.
        """
        axes = range(self.dim)
        if axis is None:
            return [self.lagrange(i) for i in axes]
        i = axes[axis]
        p = self.degree[i]
        ub = self.breaks(i)
        if p < 2: return ub
        x = np.linspace(-1, 1, p+1)
        x = (x + 1)/2
        u0 = ub[:-1]
        du = np.diff(ub)
        u = np.empty(len(du)*p+1, dtype='d')
        u[::p] = ub
        view = u[:-1].reshape(-1, p)
        for k in range(1, p):
            view[:,k] = u0 + du*x[k]
        return u

    def lobatto(self, axis=None):
        """
        Lobatto abscissae.
        """
        axes = range(self.dim)
        if axis is None:
            return [self.lobatto(i) for i in axes]
        i = axes[axis]
        p = self.degree[i]
        ub = self.breaks(i)
        if p < 2: return ub
        GaussLobattoRule = _igalib.iga.GaussLobattoRule
        x = GaussLobattoRule(p+1)[0]
        x = (x + 1)/2
        u0 = ub[:-1]
        du = np.diff(ub)
        u = np.empty(len(du)*p+1, dtype='d')
        u[::p] = ub
        view = u[:-1].reshape(-1, p)
        for k in range(1, p):
            view[:,k] = u0 + du*x[k]
        return u

    #

    def copy(self):
        """
        Copy a NURBS object.

        Returns a new instace of the NURBS objects with copies of the
        control points and knot vectors. Modifying the knot vector or
        control points of the returned object WILL NOT affect this
        object.

        Examples
        --------

        Create a random curve, copy the curve, change the control points,
        demonstrate that now c1 and c2 are different.

        >>> C = np.random.rand(5,2)
        >>> U = [0,0,1,2,3,4,4]
        >>> c1 = NURBS([U], C)
        >>> c2 = c1.copy()
        >>> c2.control[2,:] = [1.0,1.0,0.0,1.0]
        >>> np.allclose(c1.control, c2.control, rtol=0, atol=1.0e-15)
        False

        """
        nrb = NURBS.__new__(type(self))
        nrb._array = self.array.copy()
        nrb._knots = tuple(k.copy() for k in self.knots)
        return nrb

    def clone(self):
        """
        Clone a NURBS object.

        Returns a new instace of the NURBS objects with references to
        the control points and knot vectors of this NURBS
        object. Modifying the knot vector or control points of the
        returned object WILL affect this object.

        Examples
        --------

        Create a random curve, copy the curve, change the control points,
        demonstrate that changing c2 changes c1.

        >>> C = np.random.rand(5,2)
        >>> U = [0,0,1,2,3,4,4]
        >>> c1 = NURBS([U], C)
        >>> c2 = c1.clone()
        >>> c2.control[2,:] = [1.0,1.0,0.0,1.0]
        >>> np.allclose(c1.control, c2.control, rtol=0, atol=1.0e-15)
        True

        """
        nrb = NURBS.__new__(type(self))
        nrb._array = self.array
        nrb._knots = self.knots
        return nrb

    #

    def transform(self, trans):
        """
        Apply a scaling, rotation, or a translation to a NURBS object.

        A NURBS object can be scaled, rotated, or translated by
        applying the tranformation to the control points. To contruct
        composite transformations, consult the docstrings in
        :mod:`igakit.transform`.

        Parameters
        ----------
        trans : array_like
            a matrix or transformation which scales, rotates,
            and/or translates a NURBS object.

        """
        if not isinstance(trans, transform):
            trans = transform(trans)
        array = self.array.copy()
        array[...,:4] = trans(self.control)
        self._array = np.ascontiguousarray(array)
        return self

    def translate(self, displ, axis=None):
        t = transform().translate(displ, axis)
        return self.transform(t)

    def move(self, displ, axis=None):
        t = transform().move(displ, axis)
        return self.transform(t)

    def scale(self, scale, axis=None):
        t = transform().scale(scale, axis)
        return self.transform(t)

    def rotate(self, angle, axis=2):
        t = transform().rotate(angle, axis)
        return self.transform(t)

    #

    def transpose(self, axes=None):
        """
        Permute the axes of a NURBS object.

        Permute parametric axes with the given ordering and adjust the
        control points accordingly.

        Parameters
        ----------
        axes : sequence of ints, optional
            By default, reverse order of the axes, otherwise permute
            the axes according to the values given.

        Examples
        --------

        Create a random B-spline volume, swap the 2nd and 3rd axes,
        evaluate and check.

        >>> C = np.random.rand(4,3,2,3)
        >>> U = [0,0,0,0,1,1,1,1]
        >>> V = [0,0,0,1,1,1]
        >>> W = [0,0,1,1]
        >>> vol1 = NURBS([U,V,W], C)
        >>> vol2 = vol1.clone().transpose([0,2,1])
        >>> u = 0.25; v = 0.50; w = 0.75
        >>> xyz1 = vol1(u, v, w)
        >>> xyz2 = vol2(u, w, v)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True
        >>> vol3 = vol1.clone().transpose()
        >>> xyz3 = vol3(w,v,u)
        >>> np.allclose(xyz1, xyz3, rtol=0, atol=1e-15)
        True

        """
        if axes is None:
            axes = range(self.dim)[::-1]
            axes = tuple(axes)
        else:
            axes = tuple(axes)
            assert len(axes) == self.dim
        kaxes = list(axes)
        caxes = kaxes+[self.dim]
        array = self.array.transpose(caxes).copy()
        knots = [self.knots[i] for i in kaxes]
        #
        self._array = np.ascontiguousarray(array)
        self._knots = tuple(knots)
        return self

    def swap(self, axis1, axis2):
        """
        Interchange two parametric axes of NURBS object.

        Parameters
        ----------
        axis1 : int
            First axis.
        axis2 : int
            Second axis.

        Examples
        --------

        Create a random B-spline volume, swap the 2nd and 3rd axes,
        swap the first and last axes, evaluate and check.

        >>> C = np.random.rand(4,3,2,3)
        >>> U = [0,0,0,0,1,1,1,1]
        >>> V = [0,0,0,1,1,1]
        >>> W = [0,0,1,1]
        >>> vol1 = NURBS([U,V,W], C)
        >>> vol2 = vol1.clone().swap(1,2)
        >>> vol3 = vol1.clone().swap(0,-1)
        >>> u = 0.25; v = 0.50; w = 0.75
        >>> xyz1 = vol1(u, v, w)
        >>> xyz2 = vol2(u, w, v)
        >>> xyz3 = vol3(w, v, u)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True
        >>> np.allclose(xyz1, xyz3, rtol=0, atol=1e-15)
        True

        """
        allaxes = range(self.dim)
        ax1, ax2 = allaxes[axis1], allaxes[axis2]
        array = self.array.swapaxes(ax1, ax2).copy()
        knots = list(self.knots)
        knots[ax1], knots[ax2] = knots[ax2], knots[ax1]
        #
        self._array = np.ascontiguousarray(array)
        self._knots = tuple(knots)
        return self

    def reverse(self, *axes):
        """
        Reverse the parametric orientation of a NURBS object along a
        specified axis.

        Given an axis or axes, reverse the parametric orientation of
        the NURBS object. If no axes are given, reverses the
        parametric direction of all axes.

        Parameters
        ----------
        axes : int
            axis indices to reverse separated by commas

        Examples
        --------

        Create a curve, copy it, reverse the copy, evaluate at
        equivalent parametric points, verify the point is the same.

        >>> C = np.random.rand(6,2)
        >>> U = [0,0,0,0.25,0.75,0.75,1,1,1]
        >>> c1 = NURBS([U], C)
        >>> c2 = c1.copy()
        >>> c2 = c2.reverse()
        >>> u = 0.3
        >>> np.allclose(c1(u), c2(1.0-u), rtol=0, atol=1.0e-15)
        True

        """
        def CntRev(C, axis):
            dim = C.ndim - 1
            index = [slice(None, None, None)] * dim
            index[axis] = slice(None, None, -1)
            return C[tuple(index)]
        def KntRev(p, U):
            U0, U1 = U[p], U[-p-1]
            return (U1+U0)-U[::-1]
        allaxes = range(self.dim)
        if not axes: axes = allaxes
        array = self.array
        knots = list(self.knots)
        degree = self.degree
        for axis in axes:
            axis = allaxes[axis]
            array = CntRev(array, axis)
            knots[axis] = KntRev(degree[axis], knots[axis])
        #
        self._array = np.ascontiguousarray(array)
        self._knots = tuple(knots)
        return self

    #

    def remap(self, axis, start, end):
        """
        Linear reparametrization of knot vectors.

        Parameters
        ----------
        axis  : int
        start : float or None
        end   : float or None


        Examples
        --------

        >>> C = np.random.rand(6,3)
        >>> U = [0,0,0,0.25,0.75,0.75,1,1,1]
        >>> c0 = NURBS([U], C)
        >>> v0 = c0([0,0.5,1])
        >>> c1 = c0.copy().remap(0, -2, 2)
        >>> c1.knots[0].tolist()
        [-2.0, -2.0, -2.0, -1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        >>> v1 = c1([-2,0.0,2])
        >>> np.allclose(v0, v1, rtol=0, atol=1e-15)
        True
        >>> c2 = c0.copy().remap(0, None, 2)
        >>> c2.knots[0].tolist()
        [0.0, 0.0, 0.0, 0.5, 1.5, 1.5, 2.0, 2.0, 2.0]
        >>> v2 = c2([0,1.0,2])
        >>> np.allclose(v0, v2, rtol=0, atol=1e-15)
        True
        >>> c3 = c0.copy().remap(0, -1, None)
        >>> c3.knots[0].tolist()
        [-1.0, -1.0, -1.0, -0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
        >>> v3 = c3([-1,0.0,1])
        >>> np.allclose(v0, v3, rtol=0, atol=1e-15)
        True

        """
        axis = range(self.dim)[axis]
        p = self.degree[axis]
        U = self.knots[axis]
        a, b = U[p], U[-p-1]
        c, d = start, end
        if c is None: c = a
        if d is None: d = b
        c, d = float(c), float(d)
        if (a == c) and (b == d): return self
        #
        assert c < d
        U = U.copy()
        U -= a
        U *= (d-c)/(b-a)
        U += c
        knots = list(self.knots)
        knots[axis] = U
        #
        self._knots = tuple(knots)
        return self

    def insert(self, axis, value, times=1):
        """
        Insert a single knot value multiple times.

        Parameters
        ----------
        axis : int
        value: float
        times: int

        Examples
        --------

        Create a random curve, insert knots, check error:

        >>> C = np.random.rand(5,3)
        >>> U = [0,0,0,0,0.5,1,1,1,1]
        >>> c1 = NURBS([U], C)
        >>> c2 = c1.clone().insert(0, 0.25)
        >>> c3 = c2.clone().insert(0, 0.50, 2)
        >>> c4 = c3.clone().insert(0, 0.75, 3)
        >>> u = np.linspace(0,1,100)
        >>> xyz1 = c1(u)
        >>> xyz2 = c2(u)
        >>> xyz3 = c3(u)
        >>> xyz4 = c4(u)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True
        >>> np.allclose(xyz1, xyz3, rtol=0, atol=1e-15)
        True
        >>> np.allclose(xyz1, xyz4, rtol=0, atol=1e-15)
        True

        Create a random surface, insert knots, check error:

        >>> C = np.random.rand(4,3,3)
        >>> U = [0,0,0,0,1,1,1,1]; V = [0,0,0,1,1,1]
        >>> s1 = NURBS([U,V], C)
        >>> s1.shape
        (4, 3)
        >>> s2 = s1.clone().insert(0, 0.25).insert(1, 0.75, 2)
        >>> s2.shape
        (5, 5)
        >>> u = v = np.linspace(0,1,100)
        >>> xyz1 = s1(u, v)
        >>> xyz2 = s2(u, v)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True

        """
        axis = range(self.dim)[axis]
        array = self.array.view()
        knots = list(self.knots)
        p = self.degree[axis]
        U = knots[axis]
        assert U[p] <= value <= U[-p-1]
        #
        mult = _bsp.FindMult(p, U, value)
        mult = min(mult, p)
        if times is None: times = p-mult
        assert times + mult <= p
        if times == 0: return self
        #
        InsertKnot = _bsp.InsertKnot
        Pw = np.rollaxis(array, axis, 0)
        shape = Pw.shape
        Pw = Pw.reshape((shape[0], -1))
        V, Qw = InsertKnot(p, U, Pw, value, times)
        Qw.shape = (Qw.shape[0], ) + shape[1:]
        array = np.rollaxis(Qw, 0, axis+1)
        knots[axis] = V
        #
        self._array = np.ascontiguousarray(array)
        self._knots = tuple(knots)
        return self

    def remove(self, axis, value, times=1, deviation=1.0e-9):
        r"""
        Remove a single knot value multiple times.

        Parameters
        ----------
        axis : int
        value: float
        times: int
        deviation: float

        Examples
        --------

        Create a random curve, insert knots,
        remove knots, check error:

        >>> C = np.random.rand(5,3)
        >>> U = [0,0,0,0,0.5,1,1,1,1]
        >>> c1 = NURBS([U], C)
        >>> c1.shape
        (5,)
        >>> c2 = c1.clone().insert(0, 0.25) \
        ...                .remove(0, 0.25) \
        ...                .remove(0, 0.25)
        >>> c2.shape
        (5,)
        >>> c3 = c2.clone().insert(0, 0.50, 2) \
        ...                .remove(0, 0.50, 2)
        >>> c3.shape
        (5,)
        >>> c4 = c3.clone().insert(0, 0.75, 3) \
        ...                .remove(0, 0.75, 1) \
        ...                .remove(0, 0.75, 2)
        >>> c3.shape
        (5,)
        >>> u = np.linspace(0,1,100)
        >>> xyz1 = c1(u)
        >>> xyz2 = c2(u)
        >>> xyz3 = c3(u)
        >>> xyz4 = c4(u)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True
        >>> np.allclose(xyz1, xyz3, rtol=0, atol=1e-15)
        True
        >>> np.allclose(xyz1, xyz4, rtol=0, atol=1e-15)
        True

        >>> c0 = c1.remove(0, 0).remove(0, 1)
        >>> np.allclose(c1.control, c1.control, rtol=0, atol=1e-15)
        True

        """
        axis = range(self.dim)[axis]
        array = self.array.view()
        knots = list(self.knots)
        p = self.degree[axis]
        U = knots[axis]
        assert U[p] <= value <= U[-p-1]
        #
        mult = _bsp.FindMult(p, U, value)
        mult = min(mult, p)
        if times is None: times = mult
        times = min(times, mult)
        assert times >= 0
        if times == 0: return self
        if value <= U[p]: return self
        if value >= U[-p-1]: return self
        #
        RemoveKnot = _bsp.RemoveKnot
        Pw = np.rollaxis(array, axis, 0)
        shape = Pw.shape
        Pw = Pw.reshape((shape[0], -1))
        wmin = Pw[:,-1].min()
        Pmax = []
        for i in range(Pw.shape[0]):
            Pmax.append(np.linalg.norm(Pw[i,:-1]))
        Pmax = max(Pmax)
        TOL = deviation*wmin/(1.+Pmax)
        t, V, Qw = RemoveKnot(p, U, Pw, value, TOL, times)
        if t > 0: V = V[:-t].copy()
        if t > 0: Qw = Qw[:-t,:].copy()
        Qw.shape = (Qw.shape[0], ) + shape[1:]
        array = np.rollaxis(Qw, 0, axis+1)
        knots[axis] = V
        #
        self._array = np.ascontiguousarray(array)
        self._knots = tuple(knots)
        return self

    def clamp(self, axis, side=None):
        """

        Examples
        --------

        Create a random curve, unclamp, check error:

        >>> C = np.random.rand(3,3)
        >>> U = [0,0,0,1,1,1]
        >>> c1 = NURBS([U], C).unclamp(0)
        >>> c1.knots[0].tolist()
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        >>> c2 = c1.clone().clamp(0)
        >>> c2.knots[0].tolist()
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        >>> C = np.random.rand(4,4)
        >>> U = [0,0,0,0.5,1,1,1]
        >>> c1 = NURBS([U], C).unclamp(0)
        >>> c1.knots[0].tolist()
        [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        >>> c2 = c1.clone().clamp(0, side=0)
        >>> c2.knots[0].tolist()
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0]
        >>> c3 = c1.clone().clamp(0, side=1)
        >>> c3.knots[0].tolist()
        [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0, 1.0]

        """
        axis = range(self.dim)[axis]
        array = self.array
        knots = list(self.knots)
        p = self.degree[axis]
        U = knots[axis]
        #
        l, r = True, True
        if side is not None:
            assert side in (0, 1)
            if side == 0: r = False
            else:         l = False
        #
        Clamp = _bsp.Clamp
        Pw = np.rollaxis(array, axis, 0).copy()
        shape = Pw.shape
        Pw.shape = (shape[0], -1)
        V, Qw = Clamp(p, U, Pw, l, r)
        Qw.shape = shape
        array = np.rollaxis(Qw, 0, axis+1)
        knots[axis] = V
        #
        self._array = np.ascontiguousarray(array)
        self._knots = tuple(knots)
        return self

    def unclamp(self, axis, side=None, continuity=None):
        """

        Examples
        --------

        Create a random curve, unclamp, check error:

        >>> C = np.random.rand(3,3)
        >>> U = [0,0,0,1,1,1]
        >>> c1 = NURBS([U], C)
        >>> c1.knots[0].tolist()
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        >>> c2 = c1.clone().unclamp(0)
        >>> c2.knots[0].tolist()
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        >>> u = np.linspace(0,1,100)
        >>> xyz1 = c1(u)
        >>> xyz2 = c2(u)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True

        >>> c3 = c1.clone().unclamp(0,side=0)
        >>> c3.knots[0].tolist()
        [-2.0, -1.0, 0.0, 1.0, 1.0, 1.0]
        >>> u = np.linspace(0,1,100)
        >>> xyz1 = c1(u)
        >>> xyz2 = c2(u)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True

        >>> c4 = c1.clone().unclamp(0, side=1)
        >>> c4.knots[0].tolist()
        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
        >>> u = np.linspace(0,1,100)
        >>> xyz1 = c1(u)
        >>> xyz2 = c2(u)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True

        """
        axis = range(self.dim)[axis]
        array = self.array
        knots = list(self.knots)
        p = self.degree[axis]
        U = knots[axis]
        #
        l, r = True, True
        if side is not None:
            assert side in (0, 1)
            if side == 0: r = False
            else:         l = False
        C = p - 1
        if continuity is not None:
            assert 0 <= continuity <= p-1
            C = continuity
        #
        Unclamp = _bsp.Unclamp
        Pw = np.rollaxis(array, axis, 0).copy()
        shape = Pw.shape
        Pw.shape = (shape[0], -1)
        V, Qw = Unclamp(p, U, Pw, C, l, r)
        Qw.shape = shape
        array = np.rollaxis(Qw, 0, axis+1)
        knots[axis] = V
        #
        self._array = np.ascontiguousarray(array)
        self._knots = tuple(knots)
        return self

    def refine(self, axis, values):
        """
        Knot refine a NURBS object.

        Given a list of knots to insert in a parameter direction,
        refine the curve by knot refinement. The routine operates
        on the NURBS object in-place and returns the object.

        Parameters
        ----------
        axis : int
            Parameter direction to refine
        values : float or array_like
            Knots to insert

        Examples
        --------

        Create a random surface, knot refine, check error:

        >>> C = np.random.rand(4,3,3)
        >>> U = [0,0,0,0,1,1,1,1]; V = [0,0,0,1,1,1]
        >>> s1 = NURBS([U,V], C)
        >>> s1.shape
        (4, 3)
        >>> u = [0.25, 0.50, 0.75, 0.75]
        >>> v = [0.33, 0.33, 0.67, 0.67]
        >>> s2 = s1.clone().refine(0, u).refine(1, v)
        >>> s3 = s1.clone().refine(0, u).refine(1, u)
        >>> s2.shape
        (8, 7)
        >>> s3.shape
        (8, 7)
        >>> u = v = np.linspace(0,1,100)
        >>> xyz1 = s1(u, v)
        >>> xyz2 = s2(u, v)
        >>> xyz3 = s3(u, v)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True
        >>> np.allclose(xyz1, xyz3, rtol=0, atol=1e-15)
        True

        """
        def Arg(p, U, u):
            u.sort(kind='heapsort')
            assert u[ 0] >= U[p]
            assert u[-1] <= U[-p-1]
            tmp = np.concatenate((u, U[1:-1]))
            _, i = _np_unique(tmp, return_inverse=True)
            mult = np.bincount(i)
            assert mult.max() <= p
            return u
        #
        if values is None: return self
        values = np.asarray(values, dtype='d').ravel()
        if values.size == 0: return self
        axis = range(self.dim)[axis]
        p = self.degree[axis]
        U = self.knots[axis]
        u = Arg(p, U, values)
        array = self.array.view()
        knots = list(self.knots)
        #
        RefineKnotVector = _bsp.RefineKnotVector
        Pw = np.rollaxis(array, axis, 0)
        shape = Pw.shape
        Pw = Pw.reshape((shape[0], -1))
        V, Qw = RefineKnotVector(p, U, Pw, u)
        Qw.shape = Qw.shape[:1] + shape[1:]
        array = np.rollaxis(Qw, 0, axis+1)
        knots[axis] = V
        #
        self._array = np.ascontiguousarray(array)
        self._knots = tuple(knots)
        return self

    def elevate(self, axis, times=1):
        """
        Degree elevate a NURBS object.

        Given a polynomial degree to elevate in a parameter
        direction, degree-elevate the curve. The routine operates
        on the NURBS object in-place and returns the object.

        Parameters
        ----------
        axis : int
            Parameter direction to degree-elevate
        times : int, optional
            Polynomial order to elevate

        Examples
        --------

        Create a random curve, degree elevate, check error:

        >>> C = np.random.rand(3,3)
        >>> U = [0,0,0,1,1,1]
        >>> c1 = NURBS([U], C)
        >>> c1.degree
        (2,)
        >>> c2 = c1.clone().elevate(0, 2)
        >>> c2.degree
        (4,)
        >>> u = np.linspace(0,1,100)
        >>> xyz1 = c1(u)
        >>> xyz2 = c2(u)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True

        Create a random surface, degree elevate, check error:

        >>> C = np.random.rand(3,3,3)
        >>> U = [0,0,0,1,1,1]
        >>> V = [0,0,0.5,1,1]
        >>> s1 = NURBS([U,V], C)
        >>> s1.degree
        (2, 1)
        >>> s2 = s1.clone().elevate(0, 1).elevate(1, 1)
        >>> s2.degree
        (3, 2)
        >>> u = v = np.linspace(0,1,100)
        >>> xyz1 = s1(u, v)
        >>> xyz2 = s2(u, v)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True

        """
        #
        if times is None: return self
        times = int(times)
        assert times >= 0
        if times == 0: return self
        axis = range(self.dim)[axis]
        p = self.degree[axis]
        U = self.knots[axis]
        t = times
        array = self.array.view()
        knots = list(self.knots)
        #
        DegreeElevate = _bsp.DegreeElevate
        Pw = np.rollaxis(array, axis, 0)
        shape = Pw.shape
        Pw = Pw.reshape((shape[0], -1))
        V, Qw = DegreeElevate(p, U, Pw, t)
        Qw.shape = Qw.shape[:1] + shape[1:]
        array = np.rollaxis(Qw, 0, axis+1)
        knots[axis] = V
        #
        self._array = np.ascontiguousarray(array)
        self._knots = tuple(knots)
        return self

    #

    def slice(self, axis, start, end):
        """
        Parameters
        ----------
        axis: int
        start, end: float

        Examples
        --------

        Create a random curve, slice, check error:

        >>> C = np.random.rand(5,3)
        >>> U = [0,0,0,0,0.5,1,1,1,1]
        >>> crv = NURBS([U], C)
        >>> sub = crv.slice(0,0.5,0.75)
        >>> u = np.linspace(0.5,0.75,100)
        >>> xyz1 = crv(u)
        >>> xyz2 = sub(u)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=1e-15)
        True

        Create a random surface, slice along first axis,
        check error:

        >>> C = np.random.rand(5,4,3)
        >>> U = [0,0,0,0,0.5,1,1,1,1]; V = U[1:-1];
        >>> srf = NURBS([U,V], C)
        >>> sub = srf.slice(0,1./3,2./3)
        >>> u = np.linspace(1./3,2./3,100)
        >>> v = np.linspace(0,1,100)
        >>> xyz1 = srf(u, v)
        >>> xyz2 = sub(u, v)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=2e-15)
        True

        Create a random volume, slice along first and second axes,
        check error:

        >>> C = np.random.rand(4,3,2,3)
        >>> U = [0,0,0,0,1,1,1,1]; V = U[1:-1]; W = V[1:-1];
        >>> vol = NURBS([U,V,W], C)
        >>> sub = vol.slice(0,1./3,2./3).slice(1,0.25,0.75)
        >>> sub = sub.slice(0,1./3,2./3) # no-op
        >>> sub = sub.slice(1,0.25,0.75) # no-op
        >>> sub = sub.slice(2,0,1)       # no-op
        >>> u = np.linspace(1./3,2./3,100)
        >>> v = np.linspace(0.25,0.75,100)
        >>> w = np.linspace(0,1,100)
        >>> xyz1 = vol(u, v, w)
        >>> xyz2 = sub(u, v, w)
        >>> np.allclose(xyz1, xyz2, rtol=0, atol=2e-15)
        True

        """
        dim = self.dim
        axis = range(dim)[axis]
        p = self.degree[axis]
        U = self.knots[axis]
        if start is None:
            start = U[p]
        if end is None:
            end = U[-p-1]
        start = float(start)
        end = float(end)
        assert start >= U[p]
        assert end <= U[-p-1]
        assert start < end
        #
        FindSpanMult = _bsp.FindSpanMult
        u0 = start
        k0, s0 = FindSpanMult(p,U,u0)
        t0 = max(0, p-s0)
        u1 = end
        k1, s1 = FindSpanMult(p,U,u1)
        t1 = max(0, p-s1)
        u = np.repeat([u0,u1],[t0,t1]).astype('d')
        #
        nrb = self.clone()
        nrb.refine(axis, u)
        array = nrb.array
        knots = nrb.knots
        #
        U = knots[axis]
        i0 = U.searchsorted(u0, 'r') - 1
        i1 = U.searchsorted(u1, 'l')
        index = [slice(None)] * dim
        index[axis] = slice(i0-p, i1)
        array = array[tuple(index)].copy()
        Ul = U[i0].repeat(p)
        Uc = U[i0:i1+1]
        Ur = U[i1].repeat(p)
        knots = list(nrb.knots)
        knots[axis] = np.concatenate([Ul, Uc, Ur])
        #
        nrb = NURBS.__new__(type(self))
        nrb._array = np.ascontiguousarray(array)
        nrb._knots = tuple(knots)
        return nrb

    def extract(self, axis, value):
        """
        Extract lower dimensional NURBS object.

        Parameters
        ----------
        axis: int
            index of axis along which to extract.
        value: float
            parametric value from which to extract.

        Examples
        --------

        Create a random volume, extract a surface along the 3rd
        parametric direction at w=0.3, further extract a curve from
        the surface along the 1st parametric direction at u=0.5,
        compare evaluations at equivalent points.

        >>> C = np.random.rand(4,3,2,3)
        >>> U = [0,0,0,0,1,1,1,1]; V = U[1:-1]; W = V[1:-1];
        >>> u = 0.5; v = 0.75; w = 0.3;
        >>> vol = NURBS([U,V,W], C)
        >>> vol.shape
        (4, 3, 2)
        >>> vol.degree
        (3, 2, 1)
        >>> srf = vol.extract(2,w)
        >>> srf.shape
        (4, 3)
        >>> srf.degree
        (3, 2)
        >>> crv = srf.extract(1,v)
        >>> crv.shape
        (4,)
        >>> crv.degree
        (3,)
        >>> pnt = crv.extract(0,u)
        >>> pnt.shape
        ()
        >>> pnt.degree
        ()
        >>> p1 = vol(u, v, w)
        >>> p2 = srf(u, v)
        >>> p3 = crv(u)
        >>> p4 = pnt()
        >>> np.allclose(p1, p2, rtol=0, atol=1e-15)
        True
        >>> np.allclose(p1, p3, rtol=0, atol=1e-15)
        True
        >>> np.allclose(p1, p4, rtol=0, atol=1e-15)
        True
        """
        axis = range(self.dim)[axis]
        p = self.degree[axis]
        U = self.knots[axis]
        u = float(value)
        assert u >= U[p]
        assert u <= U[-p-1]
        array = self.array.view()
        knots = list(self.knots)
        #
        Extract = _bsp.Extract
        Pw = np.rollaxis(array, axis, 0)
        shape = Pw.shape
        Pw = Pw.reshape((shape[0], -1))
        Cw = Extract(p,U,Pw,u)
        array = Cw.reshape(shape[1:])
        del knots[axis]
        #
        nrb = NURBS.__new__(NURBS)
        nrb._array = np.ascontiguousarray(array)
        nrb._knots = tuple(knots)
        return nrb

    def boundary(self, axis, side):
        """
        Extract the boundary of a NURBS object along the specified
        axis and side.

        Parameters
        ----------
        axis : int
            index of axis along which to extract boundary.
        side : int
            side of axis from which to extract the boundary.

        """
        assert side in (0, 1)
        axis = range(self.dim)[axis]
        p = self.degree[axis]
        U = self.knots[axis]
        if side == 0:
            value = U[p]
        else:
            value = U[-p-1]
        return self.extract(axis, value)

    #

    def __call__(self, u=None, v=None, w=None, fields=False):
        """
        Evaluate the NURBS object at the given parametric values.

        Parameters
        ----------
        u, v, w : float or array_like, optional
        fields : bool or array_like, optional

        Examples
        --------

        >>> C = [[-1,0],[0,1],[1,0]]
        >>> U = [0,0,0,1,1,1]
        >>> crv = NURBS([U], C)
        >>> crv(0.5).tolist()
        [0.0, 0.5, 0.0]
        >>> crv([0.5]).tolist()
        [[0.0, 0.5, 0.0]]
        >>> crv([0,1]).tolist()
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        """
        def Arg(p, U, u):
            u = np.asarray(u, dtype='d')
            assert u.min() >= U[p]
            assert u.max() <= U[-p-1]
            return u
        #
        dim = self.dim
        uvw = [u,v,w][:dim]
        for i, a in enumerate(uvw):
            if a is None:
                uvw[i] = self.greville(i)
            else:
                U = self.knots[i]
                p = self.degree[i]
                uvw[i] = Arg(p, U, a)
        #
        if fields is True:
            array = self.array
        elif fields is False:
            array = np.ascontiguousarray(self.control)
            fields = False
        else:
            F = np.asarray(fields, dtype='d')
            fields = True
            shape = self.shape
            if F.shape == shape:
                F = F[...,np.newaxis]
            else:
                assert F.ndim-1 == len(shape)
                assert F.shape[:-1] == shape
            Cw = self.control
            w = Cw[...,3,np.newaxis]
            CwF = np.concatenate([Cw, F*w], axis=-1)
            array = np.ascontiguousarray(CwF)
        #
        arglist = []
        for p, U in zip(self.degree, self.knots):
            arglist.extend([p, U])
        arglist.append(array)
        arglist.extend(uvw)
        #
        Evaluate = getattr(_bsp, 'Evaluate%d' % dim)
        CwF = Evaluate(*arglist)
        w = CwF[...,3,np.newaxis]
        C = CwF[...,:3] / w
        if fields:
            F = CwF[...,4:] / w
        else:
            F = None
        #
        shape = list(C.shape[:-1])
        remove = [i for (i, a) in enumerate(uvw) if not a.ndim]
        for i in reversed(remove): del shape[i]
        C.shape = shape + [-1]
        if fields:
            F.shape = shape + [-1]
        #
        if fields:
            return C, F
        else:
            return C

    def evaluate(self, fields=None, u=None, v=None, w=None):
        """
        Parameters
        ----------
        fields : array_like, optional
        u, v, w : float or array_like

        """
        def Arg(p, U, u):
            u = np.asarray(u, dtype='d')
            assert u.min() >= U[p]
            assert u.max() <= U[-p-1]
            return u
        #
        dim = self.dim
        uvw = [u,v,w][:dim]
        for i, a in enumerate(uvw):
            if a is None:
                uvw[i] = self.greville(i)
            else:
                U = self.knots[i]
                p = self.degree[i]
                uvw[i] = Arg(p, U, a)
        #
        squeeze = False
        if fields is None:
            wF = self.array[...,3:]
        else:
            F = np.asarray(fields, dtype='d')
            shape = self.shape
            if F.shape == shape:
                F = F[...,np.newaxis]
                squeeze = True
            else:
                assert F.ndim-1 == len(shape)
                assert F.shape[:-1] == shape
            w = self.weights[...,np.newaxis]
            wF = np.concatenate([w, F*w], axis=-1)
            wF = np.ascontiguousarray(wF)
        #
        arglist = []
        for p, U in zip(self.degree, self.knots):
            arglist.extend([p, U])
        arglist.append(wF)
        arglist.extend(uvw)
        #
        Evaluate = getattr(_bsp, 'Evaluate%d' % dim)
        wF = Evaluate(*arglist)
        w = wF[...,0,np.newaxis]
        F = wF[...,1:] / w
        #
        shape = list(F.shape)
        if squeeze: del shape[dim]
        remove = [i for (i, a) in enumerate(uvw) if not a.ndim]
        for i in reversed(remove): del shape[i]
        F.shape = shape
        #
        return F

    def gradient(self, fields=None, u=None, v=None, w=None,
                 mapped=False):
        """

        Parameters
        ----------
        u, v, w : float or array_like, optional
        fields : array_like, optional
        mapped : bool, optional

        Examples
        --------

        """
        def Arg(p, U, u):
            u = np.asarray(u, dtype='d')
            assert u.min() >= U[p]
            assert u.max() <= U[-p-1]
            return u
        #
        dim = self.dim
        uvw = [u,v,w][:dim]
        for i, a in enumerate(uvw):
            if a is None:
                uvw[i] = self.greville(i)
            else:
                U = self.knots[i]
                p = self.degree[i]
                uvw[i] = Arg(p, U, a)
        #
        if fields is None:
            fields = self.fields
        if fields is None:
            fields = self.points
        F = np.asarray(fields, dtype='d')
        shape = self.shape
        if F.shape == shape:
            F = F[...,np.newaxis]
            squeeze = True
        else:
            assert F.ndim-1 == len(shape)
            assert F.shape[:-1] == shape
            squeeze = False
        #
        m  = int(bool(mapped))
        Cw = self.control
        Cw = np.ascontiguousarray(Cw)
        F  = np.ascontiguousarray(F)
        #
        arglist = []
        arglist.append(m)
        for p, U in zip(self.degree, self.knots):
            arglist.extend([p, U])
        arglist.append(Cw)
        arglist.append(F)
        arglist.extend(uvw)
        #
        Gradient = getattr(_bsp, 'Gradient%d' % self.dim)
        G = Gradient(*arglist)
        #
        shape = list(G.shape)
        if squeeze: del shape[dim]
        remove = [i for (i, a) in enumerate(uvw) if not a.ndim]
        for i in reversed(remove): del shape[i]
        G.shape = shape
        #
        return G

    def hessian(self, fields=None, u=None, v=None, w=None,
                mapped=False):
        """

        Parameters
        ----------
        u, v, w : float or array_like, optional
        fields : array_like, optional
        mapped : bool, optional

        Examples
        --------

        """
        def Arg(p, U, u):
            u = np.asarray(u, dtype='d')
            assert u.min() >= U[p]
            assert u.max() <= U[-p-1]
            return u
        #
        dim = self.dim
        uvw = [u,v,w][:dim]
        for i, a in enumerate(uvw):
            if a is None:
                uvw[i] = self.greville(i)
            else:
                U = self.knots[i]
                p = self.degree[i]
                uvw[i] = Arg(p, U, a)
        #
        if fields is None:
            fields = self.fields
        if fields is None:
            fields = self.points
        F = np.asarray(fields, dtype='d')
        shape = self.shape
        if F.shape == shape:
            F = F[...,np.newaxis]
            squeeze = True
        else:
            assert F.ndim-1 == len(shape)
            assert F.shape[:-1] == shape
            squeeze = False
        #
        m  = int(bool(mapped))
        Cw = self.control
        Cw = np.ascontiguousarray(Cw)
        F  = np.ascontiguousarray(F)
        #
        arglist = []
        arglist.append(m)
        for p, U in zip(self.degree, self.knots):
            arglist.extend([p, U])
        arglist.append(Cw)
        arglist.append(F)
        arglist.extend(uvw)
        #
        Hessian = getattr(_bsp, 'Hessian%d' % self.dim)
        H = Hessian(*arglist)
        #
        shape = list(H.shape)
        if squeeze: del shape[dim]
        remove = [i for (i, a) in enumerate(uvw) if not a.ndim]
        for i in reversed(remove): del shape[i]
        H.shape = shape
        #
        return H

    def derivative(self, order,
                   fields=None, u=None, v=None, w=None,
                   mapped=False):
        """

        Parameters
        ----------
        u, v, w : float or array_like, optional
        fields : array_like, optional
        mapped : bool, optional

        Examples
        --------

        """
        def Arg(p, U, u):
            u = np.asarray(u, dtype='d')
            assert u.min() >= U[p]
            assert u.max() <= U[-p-1]
            return u
        #
        dim = self.dim
        uvw = [u,v,w][:dim]
        for i, a in enumerate(uvw):
            if a is None:
                uvw[i] = self.greville(i)
            else:
                U = self.knots[i]
                p = self.degree[i]
                uvw[i] = Arg(p, U, a)
        #
        if fields is None:
            fields = self.fields
        if fields is None:
            fields = self.points
        F = np.asarray(fields, dtype='d')
        shape = self.shape
        if F.shape == shape:
            F = F[...,np.newaxis]
            squeeze = True
        else:
            assert F.ndim-1 == len(shape)
            assert F.shape[:-1] == shape
            squeeze = False
        #
        m  = int(bool(mapped))
        Cw = self.control
        Cw = np.ascontiguousarray(Cw)
        F  = np.ascontiguousarray(F)
        #
        arglist = []
        arglist.append(m)
        for p, U in zip(self.degree, self.knots):
            arglist.extend([p, U])
        arglist.append(Cw)
        arglist.append(F)
        arglist.extend(uvw)
        #
        assert order >= 1 and order <= 4
        fn = ['Gradient', 'Hessian', 'ThirdDer', 'FourthDer'][order-1]
        Derivative = getattr(_bsp, '%s%d' % (fn, self.dim))
        D = Derivative(*arglist)
        #
        shape = list(D.shape)
        if squeeze: del shape[dim]
        remove = [i for (i, a) in enumerate(uvw) if not a.ndim]
        for i in reversed(remove): del shape[i]
        D.shape = shape
        #
        return D

    #

    def plot(self):
        pass
