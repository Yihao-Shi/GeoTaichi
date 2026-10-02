import numpy as np
try:
    from mayavi import mlab
    from mayavi.tools import helper_functions as _helper
    from mayavi.tools import sources as _sources
    from mayavi.tools import tools as _tools
    from tvtk.api import tvtk
except ImportError:
    from enthought.mayavi import mlab
    from enthought.mayavi.tools import helper_functions as _helper
    from enthought.mayavi.tools import sources as _sources
    from enthought.mayavi.tools import tools as _tools
    from enthought.tvtk.api import tvtk
    def close_scene(scene=None, all=None):
        import copy, warnings
        from enthought.mayavi.tools.engine_manager import get_engine
        from enthought.mayavi.core.scene import Scene
        from enthought.mayavi.core.registry import registry
        if all is True:
            engine = get_engine()
            for scene in copy.copy(engine.scenes):
                engine.close_scene(scene)
            return
        if not isinstance(scene, Scene):
            engine = get_engine()
            if scene is None:
                scene = engine.current_scene
            else:
                try:
                    scene = int(scene)
                    name = 'Mayavi Scene %d' % scene
                except TypeError:
                    name = str(scene)
                for scene in engine.scenes:
                    if scene.name == name:
                        break
                else:
                    warnings.warn('Scene %s not managed by mlab' % name)
                    return
        else:
            if scene.scene is None:
                engine = registry.find_scene_engine(scene)
            else:
                engine = registry.find_scene_engine(scene.scene)
        engine.close_scene(scene)
    mlab.close = close_scene
try:
    from vtk.util import colors
except ImportError:
    class colors(object):
        black   = (0.00, 0.00, 0.00)
        grey    = (0.75, 0.75, 0.75)
        white   = (1.00, 1.00, 1.00)
        #
        red     = (1.00, 0.00, 0.00)
        green   = (0.00, 1.00, 0.00)
        blue    = (0.00, 0.00, 1.00)
        #
        yellow  = (1.00, 1.00, 0.00)
        cyan    = (0.00, 1.00, 1.00)
        magenta = (1.00, 0.00, 1.00)
        #
        orange  = (1.00, 0.50, 0.00)

myv = mlab

figure = mlab.figure
gcf = mlab.gcf
clf = mlab.clf
close = mlab.close
save = mlab.savefig
show = mlab.show

title = mlab.title
xlabel = mlab.xlabel
ylabel = mlab.ylabel
zlabel = mlab.zlabel
colorbar = mlab.colorbar

points3d = mlab.points3d
quiver3d = mlab.quiver3d

def _extract_grid_lines(LINES):
    point_list = []
    line_list  = []
    for (x, y, z) in LINES:
        assert x.ndim == y.ndim == z.ndim
        assert x.shape == y.shape, "Arrays x and y must have same shape."
        assert y.shape == z.shape, "Arrays y and z must have same shape."
        #
        xyz = [x.ravel(), y.ravel(), z.ravel()]
        points = np.column_stack(xyz).ravel()
        points.shape = (-1, 3)
        lines = np.zeros((0, 2), dtype='l')
        #
        grid = np.arange(x.size, dtype='l').reshape(x.shape)
        if x.ndim == 1:
            p0 = grid[:-1].ravel()
            p1 = grid[+1:].ravel()
            verts = [p0,p1]
            lines = np.column_stack(verts).ravel()
            lines.shape = (-1, 2)
        elif x.ndim == 2:
            p0 = grid[:-1, :-1].ravel()
            p1 = grid[+1:, :-1].ravel()
            p2 = grid[+1:, +1:].ravel()
            p3 = grid[:-1, +1:].ravel()
            verts = [p0,p1,p2,p3]
            polys = np.column_stack(verts).ravel()
            polys.shape = (-1, 4)
            lines = polys[:,[0,1,1,2,2,3,3,0]].ravel()
            lines.shape = (-1, 2)
        elif x.ndim == 3:
            p0 = grid[:-1, :-1, :-1].ravel()
            p1 = grid[+1:, :-1, :-1].ravel()
            p2 = grid[+1:, +1:, :-1].ravel()
            p3 = grid[:-1, +1:, :-1].ravel()
            p4 = grid[:-1, :-1, +1:].ravel()
            p5 = grid[+1:, :-1, +1:].ravel()
            p6 = grid[+1:, +1:, +1:].ravel()
            p7 = grid[:-1, +1:, +1:].ravel()
            verts = [p0,p1,p2,p3,
                     p4,p5,p6,p7,
                     p0,p1,p5,p4,
                     p2,p6,p7,p3,
                     p0,p3,p7,p4,
                     p1,p2,p6,p5]
            polys = np.column_stack(verts).ravel()
            polys.shape = (-1, 4)
            lines = polys[:,[0,1,1,2,2,3,3,0]].ravel()
            lines.shape = (-1, 2)
        point_list.append(points)
        line_list.append(lines)
    offset = 0
    for points, lines in zip(point_list, line_list):
        lines += offset
        offset += len(points)
    return (np.row_stack(point_list),
            np.row_stack(line_list))

def _extract_grid_polys(SURFS):
    point_list = []
    poly_list  = []
    for (x, y, z) in SURFS:
        assert x.ndim == y.ndim == z.ndim
        assert x.shape == y.shape, "Arrays x and y must have same shape."
        assert y.shape == z.shape, "Arrays y and z must have same shape."
        #
        xyz = [x.ravel(), y.ravel(), z.ravel()]
        points = np.column_stack(xyz).ravel()
        points.shape = (-1, 3)
        polys = np.zeros((0, 4), dtype='l')
        #
        grid = np.arange(x.size, dtype='l').reshape(x.shape)
        if x.ndim == 2:
            p0 = grid[:-1, :-1].ravel()
            p1 = grid[+1:, :-1].ravel()
            p2 = grid[+1:, +1:].ravel()
            p3 = grid[:-1, +1:].ravel()
            verts = [p0,p1,p2,p3]
            polys = np.column_stack(verts).ravel()
            polys.shape = (-1, 4)
        elif x.ndim == 3:
            p0 = grid[:-1, :-1, :-1].ravel()
            p1 = grid[+1:, :-1, :-1].ravel()
            p2 = grid[+1:, +1:, :-1].ravel()
            p3 = grid[:-1, +1:, :-1].ravel()
            p4 = grid[:-1, :-1, +1:].ravel()
            p5 = grid[+1:, :-1, +1:].ravel()
            p6 = grid[+1:, +1:, +1:].ravel()
            p7 = grid[:-1, +1:, +1:].ravel()
            verts = [p0,p1,p2,p3,
                     p4,p5,p6,p7,
                     p0,p1,p5,p4,
                     p2,p6,p7,p3,
                     p0,p3,p7,p4,
                     p1,p2,p6,p5]
            polys = np.column_stack(verts).ravel()
            polys.shape = (-1, 4)
        point_list.append(points)
        poly_list.append(polys)
    offset = 0
    for points, polys in zip(point_list, poly_list):
        polys += offset
        offset += len(points)
    return (np.row_stack(point_list),
            np.row_stack(poly_list))

class MLineSource(_sources.MlabSource):

    lines = _helper.List(_helper.Array, [])

    def reset(self, **traits):
        self.set(trait_change_notify=False, **traits)
        points, lines = _extract_grid_lines(self.lines)
        if self.dataset is None:
            pd = tvtk.PolyData()
        else:
            pd = self.dataset
        pd.set(lines=None, polys=None)
        pd.set(points=points)
        pd.set(lines=lines)
        pd.point_data.scalars = points[:,-1].copy()
        pd.point_data.scalars.name = 'scalars'
        self.dataset = pd

    @classmethod
    def line_source(MLineSource, lines=(), **kwargs):
        convert = _sources.convert_to_arrays
        lines = [convert((x, y, z)) for (x, y, z) in lines]
        data_source = MLineSource()
        data_source.reset(lines=lines)
        name = kwargs.pop('name', 'LineSource')
        ds = _tools.add_dataset(data_source.dataset, name, **kwargs)
        data_source.m_data = ds

class MSurfSource(_sources.MlabSource):

    surfs = _helper.List(_helper.Array, [])

    def reset(self, **traits):
        self.set(trait_change_notify=False, **traits)
        points, polys = _extract_grid_polys(self.surfs)
        if self.dataset is None:
            pd = tvtk.PolyData()
        else:
            pd = self.dataset
        pd.set(lines=None, polys=None)
        pd.set(points=points)
        pd.set(polys=polys)
        pd.point_data.scalars = points[:,-1].copy()
        pd.point_data.scalars.name = 'scalars'
        self.dataset = pd

    @classmethod
    def surf_source(MSurfSource, surfs=(), **kwargs):
        convert = _sources.convert_to_arrays
        surfs = [convert((x, y, z)) for (x, y, z) in surfs]
        data_source = MSurfSource()
        data_source.reset(surfs=surfs)
        name = kwargs.pop('name', 'SurfSource')
        ds = _tools.add_dataset(data_source.dataset, name, **kwargs)
        data_source.m_data = ds

class Line3d(_helper.Plot3d):
    lines = _helper.List(_helper.Array, [])
    _source_function = _helper.Callable(MLineSource.line_source)

class Surf3d(_helper.Mesh):
    surfs = _helper.List(_helper.Array, [])
    _source_function = _helper.Callable(MSurfSource.surf_source)

line3d = _helper.document_pipeline(Line3d())
surf3d = _helper.document_pipeline(Surf3d())

# --

_resolution = { 1:256, 2:128 }
