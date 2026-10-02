import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.cm as _cm
import mpl_toolkits.mplot3d as _mplot3d
import matplotlib.colors as _colors

class colors(object): pass
colors = colors()
colors.__dict__.update(_colors.cnames)

figure = mpl.figure
gcf = mpl.gcf
clf = mpl.clf
close = mpl.close
save = mpl.savefig
show = mpl.show

def gca(**kwargs):
    if not gcf().axes:
        gcf().add_subplot(projection='3d', **kwargs)
    return gcf().gca()

def title(*args, **kwargs):
    gca().set_title(*args, **kwargs)

def xlabel(*args, **kwargs):
    gca().set_xlabel(*args, **kwargs)

def ylabel(*args, **kwargs):
    gca().set_ylabel(*args, **kwargs)

def zlabel(*args, **kwargs):
    gca().set_zlabel(*args, **kwargs)

colorbar = mpl.colorbar

def points3d(x, y ,z, **kwargs):
    options = dict(kwargs)
    _ = options.pop('name', None)
    _ = options.pop('representation', None)
    _ = options.pop('opacity', None)
    _ = options.pop('colormap', None)
    _ = options.pop('resolution', None)
    _ = options.pop('scale_factor', None)
    _ = options.pop('scale_mode', None)
    _ = options.pop('line_width', None)
    m = options.pop('mode', 'sphere')
    if m == 'sphere':
        options['marker'] = 'o'
    if m == 'cube':
        options['marker'] = 's'
    if m == 'cone':
        options['marker'] = '^'
    ax = gca()
    x, y, z = x.T, y.T, z.T
    x, y, z = (c.ravel() for c in (x, y, z))
    pts = ax.scatter(x, y, z, s=30, **options)

def line3d(lines, **kwargs):
    options = dict(kwargs)
    _ = options.pop('name', None)
    _ = options.pop('representation', None)
    _ = options.pop('opacity', None)
    _ = options.pop('colormap', None)
    _ = options.pop('resolution', None)
    _ = options.pop('tube_radius', None)
    _ = options.pop('tube_sides', None)
    _ = options.pop('line_width', None)
    ax = gca()
    for (x, y ,z) in lines:
        x, y, z = x.T, y.T, z.T
        if x.ndim == 1:
            opts = dict(options)
            x, y, z = (c.ravel() for c in (x, y, z))
            lns = ax.plot(x, y, z, **opts)
        if x.ndim == 2:
            opts = dict(options)
            opts['rstride'] = opts['cstride'] = 1
            lns = ax.plot_wireframe(x, y, z, **opts)
        if x.ndim == 3:
            opts = dict(options)
            opts['rstride'] = opts['cstride'] = 1
            I, J, K = [np.arange(s) for s in x.shape]
            for i in I:
                X, Y, Z = (c[i,...] for c in (x, y, z))
                lns = ax.plot_wireframe(X, Y, Z, **opts)
            opts = dict(options)
            for j in J:
                for k in K:
                    X, Y, Z = (c[:,j,k].ravel() for c in (x, y, z))
                    lns = ax.plot(X, Y, Z, **opts)

def surf3d(surfs=(), **kwargs):
    options = dict(kwargs)
    _ = options.pop('name', None)
    _ = options.pop('representation', None)
    _ = options.pop('opacity', None)
    _ = options.pop('colormap', None)
    _ = options.pop('resolution', None)
    _ = options.pop('scale_factor', None)
    _ = options.pop('scale_mode', None)
    _ = options.pop('tube_radius', None)
    _ = options.pop('tube_sides', None)
    _ = options.pop('line_width', None)
    ax = gca()
    for (x, y ,z) in surfs:
        x, y, z = x.T, y.T, z.T
        if x.ndim == 2:
            opts = dict(options)
            opts['rstride'] = opts['cstride'] = 1
            if 'linewidth' not in opts:
                opts['linewidth'] = 0
            if ('color' not in opts and
                'cmap'  not in opts):
                opts['cmap'] = _cm.jet
            opts['shade'] = True
            opts['antialiased'] = True
            srf = ax.plot_surface(x, y, z, **opts)
        if x.ndim == 3:
            continue # XXX
            I, J, K = [np.arange(s) for s in x.shape]
            opts = dict(options)
            opts['rstride'] = opts['cstride'] = 1
            for i in I:
                X, Y, Z = (c[i,...] for c in (x, y, z))
                ax.plot_wireframe(X, Y, Z, **opts)
            opts = dict(options)
            for j in J:
                for k in K:
                    X, Y, Z = (c[:,j,k].ravel() for c in (x, y, z))
                    ax.plot(X, Y, Z, **opts)

_resolution = { 1:128, 2:16 }
