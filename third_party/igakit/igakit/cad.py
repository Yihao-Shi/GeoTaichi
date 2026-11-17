import numpy as np
from igakit.nurbs import NURBS
from igakit.transform import transform

# -----

Pi = np.pi
radians = np.radians
degrees = np.degrees
try:
    deg2rad = np.deg2rad
    rad2deg = np.rad2deg
except AttributeError:
    deg2rad = np.radians
    rad2deg = np.degrees

try:
    _np_unique = np.unique
except AttributeError:
    try:
        _np_unique = np.lib.arraysetops.unique
    except AttributeError:
        _np_unique = np.unique1d
try:
    _np_in1d = np.in1d
except AttributeError:
    try:
        _np_in1d = np.lib.arraysetops.in1d
    except AttributeError:
        _np_in1d = np.setmember1d

# -----

def line(p0=(0,0), p1=(1,0)):
    """
    p0         p1
    o-----------o
        +--> u
    """
    p0 = np.asarray(p0, dtype='d')
    p1 = np.asarray(p1, dtype='d')
    points = np.zeros((2,3), dtype='d')
    points[0,:p0.size] = p0
    points[1,:p1.size] = p1
    knots = [0,0,1,1]
    return NURBS([knots], points)

def circle(radius=1, center=None, angle=None):
    """
    Construct a NURBS circular arc or full circle

    Parameters
    ----------
    radius : float, optional
    center : array_like, optional
    angle : float or 2-tuple of floats, optional

    Examples
    --------

    >>> crv = circle()
    >>> crv.shape
    (9,)
    >>> P = crv([0, 0.25, 0.5, 0.75, 1])
    >>> assert np.allclose(P[0], ( 1,  0, 0))
    >>> assert np.allclose(P[1], ( 0,  1, 0))
    >>> assert np.allclose(P[2], (-1,  0, 0))
    >>> assert np.allclose(P[3], ( 0, -1, 0))
    >>> assert np.allclose(P[4], ( 1,  0, 0))

    >>> crv = circle(angle=3*Pi/2)
    >>> crv.shape
    (7,)
    >>> P = crv([0, 1/3., 2/3., 1])
    >>> assert np.allclose(P[0], ( 1,  0, 0))
    >>> assert np.allclose(P[1], ( 0,  1, 0))
    >>> assert np.allclose(P[2], (-1,  0, 0))
    >>> assert np.allclose(P[3], ( 0, -1, 0))

    >>> crv = circle(radius=2, center=(1,1), angle=(Pi/2,-Pi/2))
    >>> crv.shape
    (5,)
    >>> P = crv([0, 0.5, 1])
    >>> assert np.allclose(P[0], (1,  3, 0))
    >>> assert np.allclose(P[1], (3,  1, 0))
    >>> assert np.allclose(P[2], (1, -1, 0))

    >>> crv = circle(radius=3, center=2, angle=Pi/2)
    >>> crv.shape
    (3,)
    >>> P = crv([0, 1])
    >>> assert np.allclose(P[0], ( 5, 0, 0))
    >>> assert np.allclose(P[1], ( 2, 3, 0))

    """
    if angle is None:
        # Full circle, 4 knot spans, 9 control points
        spans = 4
        Cw = np.zeros((9,4), dtype='d')
        Cw[:,:2] = [[ 1, 0], [ 1, 1], [ 0, 1],
                    [-1, 1], [-1, 0], [-1,-1],
                    [ 0,-1], [ 1,-1], [ 1, 0]]
        Cw[:,:2] *= radius
        wm = np.sqrt(2)/2
        Cw[:,3] = 1; Cw[1::2,:] *= wm
    else:
        Pi = np.pi # inline numpy.pi
        # Determine start and end angles
        if isinstance(angle, (tuple, list)):
            start, end = angle
            if start is None: start = 0
            if end is None: end = 2*Pi
        else:
            start, end = 0, angle
        # Compute sweep and number knot spans
        sweep = end - start
        quadrants = (0.0, Pi/2, Pi, 3*Pi/2)
        spans = np.searchsorted(quadrants, abs(sweep))
        # Construct a single-segment NURBS circular arc
        # centered at the origin and bisected by +X axis
        alpha = sweep/(2*spans)
        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)
        tan_a = np.tan(alpha)
        x = radius*cos_a
        y = radius*sin_a
        wm = cos_a
        xm = x + y*tan_a
        Ca = [[    x, -y, 0,  1],
              [wm*xm,  0, 0, wm],
              [    x,  y, 0,  1]]
        # Compute control points by successive rotation
        # of the controls points in the first segment
        Cw = np.empty((2*spans+1,4), dtype='d')
        R = transform().rotate(alpha+start, 2)
        Cw[0:3,:] = R(Ca)
        if spans > 1:
            R = transform().rotate(2*alpha, 2)
            for i in range(1, spans):
                n = 2*i+1
                Cw[n:n+2,:] = R(Cw[n-2:n,:])
    # Translate control points to center
    if center is not None:
        T = transform().translate(center)
        Cw = T(Cw)
    # Compute knot vector in the range [0,1]
    a, b = 0, 1
    U = np.empty(2*(spans+1)+2, dtype='d')
    U[0], U[-1] = a, b
    U[1:-1] = np.linspace(a,b,spans+1).repeat(2)
    # Return the new NURBS object
    return NURBS([U], Cw)

def linear(points=None):
    """
    p[0]         p[1]
    o------------o
       +----> u
    """
    if points is None:
        points = np.zeros((2,2), dtype='d')
        points[0,0] = -0.5
        points[1,0] = +0.5
    else:
        points = np.asarray(points, dtype='d')
        assert points.shape[:-1] == (2,)
    knots = [0,0,1,1]
    return NURBS([knots], points)

def bilinear(points=None):
    """
    p[0,1]       p[1,1]
    o------------o
    |  v         |
    |  ^         |
    |  |         |
    |  +----> u  |
    o------------o
    p[0,0]       p[1,0]
    """
    if points is None:
        s = slice(-0.5, +0.5, 2j)
        x, y = np.ogrid[s, s]
        points = np.zeros((2,2,2), dtype='d')
        points[...,0] = x
        points[...,1] = y
    else:
        points = np.array(points, dtype='d')
        assert points.shape[:-1] == (2,2)
    knots = [0,0,1,1]
    return NURBS([knots]*2, points)

def trilinear(points=None):
    """
       p[0,1,1]     p[1,1,1]
       o------------o
      /|           /|
     / |          / |          w
    o------------o  |          ^  v
    | p[0,0,1]   | p[1,0,1]    | /
    |  |         |  |          |/
    |  o-------- | -o          +----> u
    | / p[0,1,0] | / p[1,1,0]
    |/           |/
    o------------o
    p[0,0,0]     p[1,0,0]
    """
    if points is None:
        s = slice(-0.5, +0.5, 2j)
        x, y, z = np.ogrid[s, s, s]
        points = np.zeros((2,2,2,3), dtype='d')
        points[...,0] = x
        points[...,1] = y
        points[...,2] = z
    else:
        points = np.array(points, dtype='d')
        assert points.shape[:-1] == (2,2,2)
    knots = [0,0,1,1]
    return NURBS([knots]*3, points)

# -----

def grid(shape, degree=2, continuity=None,
         limits=(0.0, 1.0), wrap=False):
    """
    Constructs a NURBS grid with equally-spaced knot vectors
    and control points built from Greville coordinates.

    Parameters
    ----------
    shape : sequence of int
    degree : int or sequence of int, optional
    continuity : int or sequence of int, optional
    limits : 2-float or sequence of 2-float, optional
    wrap : bool or sequence of bool, optional

    """
    shape = np.asarray(shape, dtype='i')
    if shape.ndim == 0:
        dim = 1
        shape.shape = (1,)
    else:
        assert shape.ndim == 1
        dim = shape.shape[0]
        assert 1 <= dim <= 3
    #
    degree = np.asarray(degree, dtype='i')
    if degree.ndim == 0:
        degree = degree.repeat(dim)
    assert degree.shape == (dim,)
    assert np.all(degree > 0)
    #
    if continuity is None:
        continuity = degree - 1
    continuity = np.asarray(continuity, dtype='i')
    if continuity.ndim == 0:
        continuity = continuity.repeat(dim)
    assert continuity.shape == (dim,)
    continuity[continuity<0] += degree[continuity<0]
    assert np.all(continuity >= 0)
    assert np.all(continuity < degree)
    #
    limits = np.asarray(limits, dtype='d')
    if limits.ndim == 1:
        assert limits.shape[0] == 2
        limits = np.row_stack([limits]*dim)
    assert limits.shape == (dim, 2)
    #
    wrap = np.asarray(wrap, dtype='?')
    if wrap.ndim == 0:
        wrap = wrap.repeat(dim)
    assert wrap.shape == (dim,)
    #
    from igakit.igalib import bsp, iga
    KnotVector = iga.KnotVector
    Greville = bsp.Greville
    knots = []; caxes = [];
    for (N, p, C, (Ui, Uf), w) \
        in zip(shape, degree, continuity, limits, wrap):
        U = KnotVector(N, p, C, Ui, Uf, w)
        X = Greville(p, U)
        knots.append(U)
        caxes.append(X)
    shape = [len(x) for x in caxes]
    control = np.zeros(shape+[4], dtype='d')
    for i, x in enumerate(caxes):
        index = [np.newaxis] * dim
        index[i] = slice(None)
        control[...,i] = x[tuple(index)]
    control[...,3] = 1.0
    #
    return NURBS(knots, control)

# -----

def compat(*nurbs, **kargs):
    """

    Parameters
    ----------
    nurbs: sequence of NURBS


    Returns
    -------
    nurbs: list of NURBS

    """
    #
    def SameBounds(nurbs, axes):
        m, n = len(nurbs), len(axes)
        bounds = np.zeros((m, n, 2), dtype='d')
        for i, nrb in enumerate(nurbs):
            degs, knts = nrb.degree, nrb.knots
            for j, axis in enumerate(axes):
                p, U = degs[axis], knts[axis]
                bounds[i,j,:] = U[p], U[-p-1]
        Umin = bounds[...,0].min(axis=0)
        Umax = bounds[...,1].max(axis=0)
        for i, nrb in enumerate(nurbs):
            for j, axis in enumerate(axes):
                a, b = Umin[j], Umax[j]
                nrb.remap(axis, a, b)
    #
    def SameDegree(nurbs, axes):
        # Ensure same degree by degree elevation
        degree = [nrb.degree for nrb in nurbs]
        degree = np.row_stack(degree)
        degree = degree[:,axes]
        degmax = degree.max(axis=0)
        elevate = degmax - degree
        for i, nrb in enumerate(nurbs):
            for j, axis in enumerate(axes):
                t = elevate[i,j]
                nrb.elevate(axis, t)
    #
    def MergeKnots(nurbs, axes):
        m, n = len(nurbs), len(axes)
        insert = np.empty((m, n), dtype=object)
        for j, axis in enumerate(axes):
            # Knot vector -> breaks & multiplicities
            breaks = []; mults = [];
            for nrb in nurbs:
                u, s = nrb.breaks(axis, mults=True)
                breaks.append(u[1:-1])
                mults .append(s[1:-1])
            # Merge breaks and multiplicities
            masks = []
            u = _np_unique(np.concatenate(breaks))
            s = np.zeros(u.size, dtype='i')
            for (ui, si) in zip(breaks, mults):
                mask = _np_in1d(u, ui)
                s[mask] = np.maximum(s[mask], si)
                masks.append(mask)
            # Compute knots to insert
            for i, (mi, si) in enumerate(zip(masks, mults)):
                t = s.copy(); t[mi] -= si
                v = np.repeat(u, t).astype('d')
                insert[i,j] = v
        # Apply knot refinement
        for i, nrb in enumerate(nurbs):
            for j, axis in enumerate(axes):
                u = insert[i,j]
                nrb.refine(axis, u)
    #
    if len(nurbs) == 1:
        if not isinstance(nurbs[0], NURBS):
            nurbs = nurbs[0]
    nurbs = [nrb.clone() for nrb in nurbs]
    if len(nurbs) < 2: return nurbs
    assert (min(nrb.dim for nrb in nurbs) ==
            max(nrb.dim for nrb in nurbs))
    #
    dim = nurbs[0].dim
    allaxes = list(range(dim))
    axes = kargs.pop('axes', None)
    assert not kargs
    if axes is None:
        axes = allaxes
    else:
        axes = np.atleast_1d(axes)
        axes = [allaxes[i] for i in axes]
    if not axes: return nurbs
    #
    SameBounds(nurbs, axes)
    SameDegree(nurbs, axes)
    MergeKnots(nurbs, axes)
    #
    return nurbs

# -----

def extrude(nrb, displ, axis=None):
    """
    Construct a NURBS surface/volume by
    extruding a NURBS curve/surface.

    Parameters
    ----------
    nrb : NURBS
    displ : array_like or float
    axis : array_like or int, optional

    Example
    -------

    >>> crv = circle()
    >>> srf = extrude(crv, displ=1, axis=2)

    >>> srf = bilinear()
    >>> vol = extrude(srf, displ=1, axis=2)

    """
    assert nrb.dim <= 2
    T = transform().translate(displ, axis)
    Cw = np.empty(nrb.shape+(2,4))
    Cw[...,0,:] = nrb.control
    Cw[...,1,:] = T(nrb.control)
    UVW = nrb.knots + ([0,0,1,1],)
    return NURBS(UVW, Cw)

def revolve(nrb, point, axis=2, angle=None):
    """
    Construct a NURBS surface/volume by
    revolving a NURBS curve/surface.

    Parameters
    ----------
    nrb : NURBS
    point : array_like
    axis : int or array_like, optional
    angle : float or 2-tuple of floats, optional

    Example
    -------

    >>> crv = line(1,2)
    >>> srf = revolve(crv, point=0, axis=2, angle=[Pi/2,2*Pi])
    >>> vol = revolve(srf, point=3, axis=1, angle=-Pi/2)

    """
    assert nrb.dim <= 2
    point = np.asarray(point, dtype='d')
    assert point.ndim in (0, 1)
    assert point.size <= 3
    axis = np.asarray(axis)
    assert axis.ndim in (0, 1)
    assert 1 <= axis.size <= 3
    if axis.ndim == 0:
        v = np.zeros(3, dtype='d')
        axis = (0,1,2)[int(axis)]
        v[axis] = 1
    else:
        v = np.zeros(3, dtype='d')
        v[:axis.size] = axis
        norm_axis = np.linalg.norm(v)
        assert norm_axis > 0
        v /= norm_axis
    # Transform the NURBS object to a new reference frame
    # (O,X,Y,Z) centered at point and z-oriented with axis
    n = [v[1], -v[0], 0]    # n = cross(v, z)
    gamma = np.arccos(v[2]) # cos_gamma = dot(v, z)
    T = transform().translate(-point).rotate(gamma, n)
    nrb = nrb.clone().transform(T)
    # Map cartesian coordinates (x,y,z) to cylindrical coordinates
    # (rho,theta,z) with theta in [0,2*pi] and precompute sines and
    # cosines of theta angles.
    Cw = nrb.control
    X, Y, Z, W = (Cw[...,i] for i in range(4))
    rho = np.hypot(X, Y)
    theta = np.arctan2(Y, X); theta[theta<0] += 2*np.pi
    sines, cosines = np.sin(theta), np.cos(theta)
    # Create a circular arc in the XY plane
    arc = circle(angle=angle)
    Aw = arc.control
    # Allocate control points and knots of the result
    Qw = np.empty(nrb.shape + arc.shape + (4,))
    UVW = nrb.knots + arc.knots
    # Loop over all control points of the NURBS object
    # to revolve taking advantage of NumPy nd-indexing
    dot = np.dot # inline numpy.dot
    zeros = np.zeros # inline numpy.zeros
    for idx in np.ndindex(nrb.shape):
        z = Z[idx]
        w = W[idx]
        r = rho[idx]
        r_sin_a = r*sines[idx]
        r_cos_a = r*cosines[idx]
        # for the sake of speed, inline
        # the transformation matrix
        # M = Rz(theta)*Tz(z)*Sxy(rho)
        M = zeros((4,4))
        M[0,0] = r_cos_a; M[0,1] = -r_sin_a
        M[1,0] = r_sin_a; M[1,1] =  r_cos_a
        M[2,3] = z
        M[3,3] = 1
        # Compute new 4D control points by transforming the
        # arc control point and tensor-product the weights
        Qi = Qw[idx]
        Qi[...] = dot(Aw, M.T)
        Qi[...,3] *= w
    # Create the new NURBS object and map
    # back to the original reference frame
    return NURBS(UVW, Qw).transform(T.invert())

def ruled(nrb1, nrb2):
    """
    Construct a ruled surface/volume
    between two NURBS curves/surfaces.

    Parameters
    ----------
    nrb1, nrb2 : NURBS

    """
    assert nrb1.dim == nrb2.dim
    assert nrb1.dim <= 2
    assert nrb2.dim <= 2
    nrb1, nrb2 = compat(nrb1, nrb2)
    Cw = np.zeros(nrb1.shape+(2,4),dtype='d')
    Cw[...,0,:] = nrb1.control
    Cw[...,1,:] = nrb2.control
    UVW = nrb1.knots + ([0,0,1,1],)
    return NURBS(UVW, Cw)

def sweep(section, trajectory):
    """
    Construct the translational sweep of a section
    curve/surface along a trajectory curve.

    S(u,v) = C(u) + T(v)

    V(u,v,w) = S(u,v) + T(w)

    Parameters
    ----------

    section : NURBS
        Section curve/surface
    trajectory : NURBS
        Trajectory curve

    """
    assert 1 <= section.dim <= 2
    assert trajectory.dim == 1
    Cs, ws = section.points, section.weights
    Ct, wt = trajectory.points, trajectory.weights
    C = Cs[...,np.newaxis,:] + Ct
    w = ws[...,np.newaxis] * wt
    UVW = section.knots + trajectory.knots
    return NURBS(UVW, (C, w))

def coons(curves):
    """
                C[1,1]
           o--------------o
           |  v           |
           |  ^           |
    C[0,0] |  |           | C[0,1]
           |  |           |
           |  +------> u  |
           o--------------o
                C[1,0]
    """
    (C00, C01), (C10, C11) = curves
    assert C00.dim == C01.dim == 1
    assert C10.dim == C11.dim == 1
    #
    (C00, C01) = compat(C00, C01)
    (C10, C11) = compat(C10, C11)
    #
    p, U = C10.degree[0], C10.knots[0]
    u0, u1 = U[p], U[-p-1]
    P = np.zeros((2,2,3), dtype='d')
    P[0,0] = C10(u0)
    P[1,0] = C10(u1)
    P[0,1] = C11(u0)
    P[1,1] = C11(u1)
    #
    q, V = C00.degree[0], C00.knots[0]
    v0, v1 = V[q], V[-q-1]
    Q = np.zeros((2,2,3), dtype='d')
    Q[0,0] = C00(v0)
    Q[0,1] = C00(v1)
    Q[1,0] = C01(v0)
    Q[1,1] = C01(v1)
    #
    assert np.allclose(P, Q, rtol=0, atol=1e-15)
    #
    R0 = ruled(C00, C01).transpose()
    R1 = ruled(C10, C11)
    B = bilinear(P)
    R0, R1, B = compat(R0, R1, B)
    control = R0.control + R1.control - B.control
    knots = B.knots
    return NURBS(knots, control)

# -----

def join(nrb1, nrb2, axis):
    """
    Join two curves/surfaces/volumes along a
    specified parametric axis.

    Parameters
    ----------
    nrb1, nrb2 : NURBS
    axis : int

    Examples
    --------

    >>> C1 = circle(radius=0.5)
    >>> C2 = circle(radius=1.0)
    >>> annulus = ruled(C1, C2)
    >>> pipe = extrude(annulus, displ=2, axis=2)
    >>> elbow = revolve(annulus, point=(1.5,0,0),
    ...                 axis=(0,-1,0), angle=Pi/2)
    >>> bentpipe = join(pipe.reverse(2), elbow, axis=2)
    """
    dim = nrb1.dim
    assert dim == nrb2.dim
    assert 0 <= axis < dim

    axes = list(range(dim))
    del axes[axis]
    nrb1, nrb2 = compat(nrb1, nrb2, axes=axes)

    nrb1 = nrb1.clamp(axis, side=1)
    nrb2 = nrb2.clamp(axis, side=0)

    p1 = nrb1.degree[axis]
    p2 = nrb2.degree[axis]
    U1 = nrb1.knots[axis]
    U2 = nrb2.knots[axis]

    u = U1[-p1-1]
    a = U2[p2]
    b = U2[-p2-1]
    nrb2.remap(axis, a+u, b+u)

    p = max(p1, p2)
    nrb1.elevate(axis, p-p1)
    nrb2.elevate(axis, p-p2)

    A1 = nrb1.array
    A2 = nrb2.array
    I1 = [slice(None)] * (dim+1)
    I2 = [slice(None)] * (dim+1)
    I1[axis] = slice(0,-1)
    I2[axis] = slice(1,None)
    Al = A1[tuple(I1)]
    Ar = A2[tuple(I2)]
    I1[axis] = -1
    I2[axis] = +0
    Ac = (A1[tuple(I1)]+A2[tuple(I2)])/2.0
    Ic = [slice(None)] * (dim+1)
    Ic[axis] = np.newaxis
    Ac = Ac[tuple(Ic)]
    A = np.concatenate([Al, Ac, Ar], axis)

    U1 = nrb1.knots[axis]
    U2 = nrb2.knots[axis]
    Ul = U1[:-p-1]
    Ur = U2[p+1:]
    Uc = [u]*p
    U = np.concatenate([Ul, Uc, Ur])
    knots = list(nrb1.knots)
    knots[axis] = U

    nrb = NURBS.__new__(type(nrb1))
    nrb._array = np.ascontiguousarray(A)
    nrb._knots = tuple(knots)
    nrb.remove(axis, u, p-1)
    return nrb

# -----

def refine(nrb, factor=None, degree=None, continuity=None):
    """
    Refine a NURBS object by degree elevation and knot insertion.

    Parameters
    ----------
    nrb : NURBS
    factor : int or sequence of int, optional
    degree : int or sequence of int, optional
    continuity : int or sequence of int, optional

    """
    def Arg(arg, defval):
        if arg is None:
            return list(defval)
        dim = len(defval)
        arg = np.asarray(arg, dtype=object)
        if arg.ndim == 0:
            return arg.repeat(dim)
        assert arg.shape[0] == dim
        for i, val in enumerate(arg):
            if val is None:
                arg[i] = defval[i]
        return arg
    #
    nrb = nrb.clone()
    dim = nrb.dim
    # clamping
    for axis in range(dim):
        nrb.clamp(axis)
    # degree elevation
    degree = Arg(degree, nrb.degree)
    degree = np.asarray(degree, dtype='i')
    assert np.all(degree > 0)
    for axis in range(dim):
        p = degree[axis]
        t = p - nrb.degree[axis]
        if t > 0:
            nrb.elevate(axis, t)
        p = nrb.degree[axis]
        degree[axis] = p
    # knot refinement
    factor = Arg(factor, [1]*dim)
    for i, fact in enumerate(factor):
        factor[i] = np.asarray(fact, dtype='i')
        assert np.all(factor[i] > 0)
    continuity = Arg(continuity, [-1]*dim)
    continuity = np.asarray(continuity, dtype='i')
    continuity[continuity<0] += degree[continuity<0]
    assert np.all(continuity >= 0)
    assert np.all(continuity < degree)
    for axis in range(dim):
        N = factor[axis]
        C = continuity[axis]
        p = nrb.degree[axis]
        # breaks & multiplicities
        u, s = nrb.breaks(axis, mults=True)
        # compute knots to insert
        assert N.ndim == 0 or N.size == u.size-1
        delta_u = np.ediff1d(u)/N
        u = u[:-1]; s = s[:-1];
        if N.ndim == 0:
            U = u[:,np.newaxis].repeat(N, axis=1)
            step = np.arange(1, N, dtype='d')
            U[:,1:] += np.outer(delta_u, step)
            S = np.empty((s.size, N), dtype='i')
            S[:,0] = (p-C-s).clip(0, None)
            S[:,1:] = p-C
            u = U.ravel()[1:]
            s = S.ravel()[1:]
        else:
            pos = np.empty(N.size+1, dtype='i')
            pos[0] = 0; np.cumsum(N, out=pos[1:])
            U = np.empty(pos[-1], dtype='d')
            S = np.empty(pos[-1], dtype='i')
            for i, n in enumerate(N):
                uu = np.arange(n, dtype='d')
                uu *= delta_u[i]; uu += u[i]
                a, b = pos[i], pos[i+1]
                U[a:b] = uu
                S[a] = max(p-C-s[i], 0)
                S[a+1:b] = p-C
            u = U[1:]
            s = S[1:]
        # insert knots
        nrb.refine(axis, u.repeat(s))
    #
    return nrb

# -----
