import numpy as np

class Plotter(object):

    def __init__(self, backend=None):
        if backend is not None:
            self.use(backend)

    # ----------

    def points(self, points, **kwargs):
        points = np.asarray(points, dtype='d')
        points = np.atleast_2d(points)
        shape = points.shape[:-1]
        dim = points.shape[-1]
        assert 1 <= dim <= 3
        if dim < 3:
            C = np.zeros(shape+(3,), dtype='d')
            C[...,0:dim] = points
        else:
            C = points
        x, y, z = C.T
        #
        options = dict(kwargs)
        if 'mode' not in options:
            options['mode'] = 'sphere'
        #
        pts = self.backend.points3d(x, y, z, **options)
        return pts

    def quiver(self, points, vectors, **kwargs):
        #
        points = np.asarray(points, dtype='d')
        points = np.atleast_2d(points)
        dim = points.shape[-1]
        assert 1 <= dim <= 3
        if dim < 3:
            shape = points.shape[:-1]
            C = np.zeros(shape+(3,), dtype='d')
            C[...,0:dim] = points
        else:
            C = points
        x, y, z = C.T
        #
        vectors = np.asarray(vectors, dtype='d')
        vectors = np.atleast_2d(vectors)
        dim = vectors.shape[-1]
        assert 1 <= dim <= 3
        if dim < 3:
            shape = vectors.shape[:-1]
            A = np.zeros(shape+(3,), dtype='d')
            A[...,0:dim] = vectors
        else:
            A = vectors
        u, v, w = A.T
        #
        options = dict(kwargs)
        if 'mode' not in options:
            options['mode'] = 'arrow'
        #
        vcs = self.backend.quiver3d(x, y, z, u, v, w, **options)
        return vcs

    # ----------

    def cpoint(self, nurbs, **kwargs):
        C = nurbs.points
        x, y, z = C.T
        #
        options = dict(kwargs)
        if 'color' not in options:
            options['color'] = (1,0,0)
        if 'mode' not in options:
            options['mode'] = 'sphere'
        #
        pts = self.backend.points3d(x, y, z, **options)
        if 'scale_factor' not in options:
            try:
                pts.glyph.glyph.scale_factor *= 0.125
            except AttributeError:
                pass
        return pts

    def cwire(self, nurbs, **kwargs):
        C = nurbs.points
        x, y, z = C.T
        #
        options = dict(kwargs)
        if 'color' not in options:
            options['color'] = (0,0,1)
        mode = options.pop('mode', 'line')
        assert mode in ('line', 'tube')
        if mode == 'line':
            options['representation'] = 'wireframe'
            options['tube_radius'] = None
        elif mode == 'tube':
            options['representation'] = 'surface'
        #
        lines = [(x, y, z)]
        grd = self.backend.line3d(lines=lines, **options)
        return grd

    def kpoint(self, nurbs, **kwargs):
        uvw = nurbs.breaks()
        C = nurbs(*uvw)
        x, y, z = C.T
        #
        options = dict(kwargs)
        if 'color' not in options:
            options['color'] = (0,1,0)
        if 'mode' not in options:
            options['mode'] = 'cube'
        #
        pts = self.backend.points3d(x, y, z, **options)
        if 'scale_factor' not in options:
            try:
                pts.glyph.glyph.scale_factor *= 0.1
            except AttributeError:
                pass
        return pts

    def kwire(self, nurbs, axes=None, **kwargs):
        if axes is None:
            axes = (0,1,2)[:nurbs.dim]
        elif not isinstance(axes, (list, tuple)):
            axes = (axes,)
        #
        uvw = nurbs.breaks()
        lines = []
        for axis in axes:
            u = uvw[axis]
            resolution = self.backend._resolution[1]
            a = np.linspace(u[0], u[-1], resolution)
            abc = list(uvw)
            abc[axis] = a
            C = nurbs(*abc)
            C = np.rollaxis(C, axis, -1)
            C = C.reshape((-1, a.size, 3))
            lines.extend(C)
        #
        options = dict(kwargs)
        if 'color' not in options:
            options['color'] = (0,1,0)
        mode = options.pop('mode', 'line')
        assert mode in ('line', 'tube')
        if mode == 'line':
            options['representation'] = 'wireframe'
            options['tube_radius'] = None
        elif mode == 'tube':
            options['representation'] = 'surface'
        #
        lines = [tuple(C.T) for C in lines]
        wire = self.backend.line3d(lines=lines, **options)
        return wire

    def ksurf(self, nurbs, axes=None, **kwargs):
        if axes is None:
            axes = (0,1,2)[:nurbs.dim]
        elif not isinstance(axes, (list, tuple)):
            axes = (axes,)
        #
        surfs = []
        for axis in axes:
            for u in nurbs.breaks(axis):
                nrb = nurbs.extract(axis, u)
                resolution = self.backend._resolution[2]
                uvw = [np.linspace(U[p], U[-p-1], resolution)
                       for (p, U) in zip(nrb.degree, nrb.knots)]
                C = nrb(*uvw)
                surfs.append(C)
        #
        options = dict(kwargs)
        if 'color' not in options:
            options['color'] = (0,1,0)
        options['representation'] = 'surface'
        #
        surfs = [tuple(C.T) for C in surfs]
        surf = self.backend.surf3d(surfs=surfs, **options)
        return surf

    # ----------

    def curve(self, nurbs, **kwargs):
        if nurbs.dim < 1: return None
        if nurbs.dim > 1:
            boundaries = []
            for axis in range(nurbs.dim):
                for side in range(2):
                    nrb = nurbs.boundary(axis, side)
                    boundaries.append(nrb)
            return [self.curve(nrb, **kwargs)
                    for nrb in boundaries]
        #
        resolution = self.backend._resolution[1]
        p, U = nurbs.degree[0], nurbs.knots[0]
        u = np.linspace(U[p], U[-p-1], resolution)
        C = nurbs(u)
        x, y, z = C.T
        #
        options = dict(kwargs)
        color = options.pop('color', (1,1,0))
        if color is not None:
            options['color'] = color
        mode = options.pop('mode', 'tube')
        assert mode in ('line', 'tube')
        if mode == 'line':
            options['representation'] = 'wireframe'
            options['tube_radius'] = None
        elif mode == 'tube':
            options['representation'] = 'surface'
        #
        lines = [(x, y, z)]
        crv = self.backend.line3d(lines=lines, **options)
        return crv

    def surface(self, nurbs, **kwargs):
        if nurbs.dim < 2: return None
        if nurbs.dim > 2:
            surfaces = []
            for axis in range(nurbs.dim):
                for side in range(2):
                    nrb = nurbs.boundary(axis, side)
                    surfaces.append(nrb)
        else:
            surfaces = [nurbs]
        #
        surfs = []
        for nrb in surfaces:
            resolution = self.backend._resolution[2]
            uvw = [np.linspace(U[p], U[-p-1], resolution)
                   for (p, U) in zip(nrb.degree, nrb.knots)]
            C = nrb(*uvw)
            surfs.append(C)
        #
        options = dict(kwargs)
        color = options.pop('color', (1,1,0))
        if color is not None:
            options['color'] = color
        options['representation'] = 'surface'
        #
        surfs = [tuple(C.T) for C in surfs]
        srf = self.backend.surf3d(surfs=surfs, **options)
        return srf

    def volume(self, nurbs, **kwargs):
        if nurbs.dim < 3: return None
        #
        options = dict(kwargs)
        color = options.pop('color', (1,1,0))
        if color is not None:
            options['color'] = color
        #
        return self.surface(nurbs, **kwargs)

    # ----------

    def cplot(self, nurbs, **kwargs):
        opts = dict(kwargs)
        pts = self.cpoint(nurbs, **opts)
        opts = dict(kwargs)
        grd = self.cwire(nurbs, **opts)
        return (pts, grd)

    def kplot(self, nurbs, **kwargs):
        opts = dict(kwargs)
        pts = self.kpoint(nurbs, **opts)
        opts = dict(kwargs)
        grd = self.kwire(nurbs, **opts)
        return (pts, grd)

    def plot(self, nurbs, **kwargs):
        if nurbs.dim == 1:
            return self.curve(nurbs, **kwargs)
        if nurbs.dim == 2:
            return self.surface(nurbs, **kwargs)
        if nurbs.dim == 3:
            return self.volume(nurbs, **kwargs)
        return None

    # ----------

    def __getattr__(self, attr):
        return getattr(self.backend, attr)

    _modules = {
        'mpl': None,
        'myv': None,
        'nul': None,
        }
    _alias = {
        'matplotlib' : 'mpl',
        'mayavi'     : 'myv',
        'null'      :  'nul',
        'none'      :  'nul',
        }
    _backend = None

    def use(self, backend):
        self.backend = backend

    def set_backend(self, backend):
        name = self._alias.get(backend, backend)
        try:
            module = self._modules[name]
        except KeyError:
            raise ValueError("unknown backend '%s'" % backend)
        if module is None:
            modname = 'igakit.plot_' + name
            module = __import__(modname, fromlist=[None])
            self._modules[name] = module
        self._backend = module

    def get_backend(self):
        if self._backend is None:
            try:
                self.use('mayavi')
            except ImportError:
                self.use('matplotlib')
        return self._backend

    backend = property(get_backend, set_backend)

    # ----------


plt = plotter = Plotter()

use   = plotter.use
plot  = plotter.plot
cplot = plotter.cplot
kplot = plotter.kplot
