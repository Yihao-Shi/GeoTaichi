import numpy as np
import pyvista as pv
from shapely.geometry import Polygon


class NurbsPlot(object):
    def __init__(self, dimension=3) -> None:
        self.dimension = dimension
        self.evalpts = []
        self.ctrlpts = []

    def append(self, evalpts, control_points):
        if isinstance(evalpts, (list, tuple)):
            evalpts = np.asarray(evalpts)
        self.evalpts.append(evalpts)
        self.ctrlpts.append(control_points)

    def initialize(self, **kwargs):
        self.display_ctrlpts = kwargs.get('ctrlpts', True)
        self.display_boundary_ctrlpts = kwargs.get('boundary_ctrlpts', False)
        self.display_evalpts = kwargs.get('evalpts', True)
        self.display_bbox = kwargs.get('bbox', False)
        self.display_legend = kwargs.get('legend', True)
        self.display_axes = kwargs.get('axes', True)
        self.display_labels = kwargs.get('labels', True)
        self.axes_equal = kwargs.get('axes_equal', True)
        self.figure_size = kwargs.get('figure_size', [10, 8])
        self.figure_dpi = kwargs.get('figure_dpi', 96)
        self.trim_size = kwargs.get('trim_size', 20)
        self.alpha = kwargs.get('alpha', 1.0)
        self.export_file = kwargs.get("export_file", False)
        self.data_point = kwargs.get('datapts', None)
        self.finalize = kwargs.get('finalize', True)
        self.bounding_volume = kwargs.get('bounding_volume', False)

        if isinstance(self.data_point, (list, tuple, np.ndarray)):
            self.data_point = np.asarray(self.data_point).reshape(-1, self.dimension)

    @staticmethod
    def set_axes_equal(ax):
        bounds = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
        ranges = [abs(bound[1] - bound[0]) for bound in bounds]
        centers = [np.mean(bound) for bound in bounds]
        radius = 0.5 * max(ranges)
        lower_limits = centers - radius
        upper_limits = centers + radius
        ax.set_xlim3d([lower_limits[0], upper_limits[0]])
        ax.set_ylim3d([lower_limits[1], upper_limits[1]])
        ax.set_zlim3d([lower_limits[2], upper_limits[2]])

    def is_notebook(self):
        return pv._wrappers.get_notebook() is not None 
    
    def PlotCurve(self):
        if self.dimension == 2:
            self.PlotCurve2D()
        elif self.dimension == 3:
            self.PlotCurve3D()
        else:
            raise ValueError("Dimension must be 2 or 3")

    def PlotCurve2D(self):
        plotter = pv.Plotter()

        # 控制点和连线
        if self.display_ctrlpts:
            for index, ctrlpt in enumerate(self.ctrlpts):
                pts = np.c_[ctrlpt[:, 0], ctrlpt[:, 1], np.zeros(ctrlpt.shape[0])]
                # 控制点散点
                ctrlpt_cloud = pv.PolyData(pts)
                print(ctrlpt_cloud)
                plotter.add_mesh(ctrlpt_cloud, color='red',
                                 point_size=10, render_points_as_spheres=True,
                                 label=f'Control points {index}')

                # 控制点连线
                if len(ctrlpt) >= 2:
                    npts = ctrlpt.shape[0]
                    lines = np.hstack(([npts], np.arange(npts))).astype(np.int64)
                    ctrlpt_line = pv.PolyData(pts, lines=lines)
                    plotter.add_mesh(ctrlpt_line, color='red', line_width=2)

        # 曲线
        if self.display_evalpts:
            for index, evalpt in enumerate(self.evalpts):
                if len(evalpt) >= 2:
                    pts = np.c_[evalpt[:, 0], evalpt[:, 1], np.zeros(evalpt.shape[0])]
                    curve_line = pv.Spline(pts, len(pts) * 10)
                    plotter.add_mesh(curve_line, color='black',
                                     line_width=3, opacity=self.alpha,
                                     label=f'NURBS Curve {index}')

        # 采样点
        if self.data_point is not None and len(self.data_point) > 0:
            pts = np.c_[self.data_point[:, 0], self.data_point[:, 1], np.zeros(len(self.data_point))]
            sample_cloud = pv.PolyData(pts)
            plotter.add_mesh(sample_cloud, color='green',
                             point_size=10, render_points_as_spheres=True,
                             label='Sampling points')

        # 坐标轴、边界和图例
        plotter.show_axes()
        plotter.show_bounds(grid='front', location='outer', all_edges=True)
        plotter.camera_position = 'xy'

        if self.display_legend:
            plotter.add_legend()

        plotter.show()


    def PlotCurve3D(self):
        """
        用 pyvista 绘制3D曲线和控制点，带坐标轴刻度
        """
        plotter = pv.Plotter()
        
        # 确保evalpts里所有曲线都是3D点
        for i, evalpt in enumerate(self.evalpts):
            if evalpt.shape[1] == 2:
                self.evalpts[i] = np.c_[evalpt, np.zeros(evalpt.shape[0])]
        
        # 画控制点
        if self.display_ctrlpts:
            for index, ctrlpt in enumerate(self.ctrlpts):
                pts = ctrlpt  # 形状 (n, 3)
                ctrlpt_cloud = pv.PolyData(pts)
                plotter.add_mesh(ctrlpt_cloud, color='red', point_size=10, render_points_as_spheres=True, label=f'Control points {index}')
                # 连线
                lines = np.hstack(([pts.shape[0]], np.arange(pts.shape[0])))
                ctrlpt_line = pv.PolyData(pts)
                ctrlpt_line.lines = lines
                plotter.add_mesh(ctrlpt_line, color='red', line_width=2, style='wireframe')
        
        # 画评估点曲线
        if self.display_evalpts:
            for index, evalpt in enumerate(self.evalpts):
                pts = evalpt
                curve_line = pv.Spline(pts, len(pts)*10)
                plotter.add_mesh(curve_line, color='black', line_width=3, opacity=self.alpha, label=f'NURBS curve {index}')
        
        # 采样点
        if self.data_point is not None:
            pts = self.data_point
            sample_cloud = pv.PolyData(pts)
            plotter.add_mesh(sample_cloud, color='green', point_size=10, render_points_as_spheres=True, label='Sampling points')
        
        # 显示坐标轴刻度
        plotter.show_axes()
        
        # 显示图例
        if self.display_legend:
            plotter.add_legend()
        
        plotter.show()

    def ensure_3d(self, points: np.ndarray) -> np.ndarray:
        """保证点云是 Nx3，如果是 Nx2 自动补一列 0"""
        if points.shape[1] == 2:
            zeros = np.zeros((points.shape[0], 1), dtype=points.dtype)
            points = np.hstack([points, zeros])
        return points

    def PlotSurface(self, surf_cmaps='viridis'):
        """
        使用 PyVista 绘制 NURBS 曲面及控制点网格。

        surf_cmaps: PyVista 支持的 colormap 名称，比如 'viridis', 'rainbow' 等。
        """
        # 创建一个PyVista绘图器
        plotter = pv.Plotter()

        # 画控制点网格（散点+线框）
        if self.display_ctrlpts:
            # ctrlpts 形状一般是 (n_u, n_v, dim)，dim一般是3
            for ctrlpts in self.ctrlpts:
                # 展平成点云
                points = self.ensure_3d(ctrlpts.reshape(-1, ctrlpts.shape[-1]))
                cloud = pv.PolyData(points)
                plotter.add_mesh(cloud, color='red', point_size=10, render_points_as_spheres=True, label="Control Points")

                # 画控制点网格线，先画u方向
                for i in range(ctrlpts.shape[0]):
                    line_pts = self.ensure_3d(ctrlpts[i, :, :])
                    # 更细致的控制可以用PolyLine，这里简化处理
                    plotter.add_mesh(pv.Spline(line_pts, 100), color='red', line_width=2)

                # 画v方向
                for j in range(ctrlpts.shape[1]):
                    line_pts = self.ensure_3d(ctrlpts[:, j, :])
                    plotter.add_mesh(pv.Spline(line_pts, 100), color='red', line_width=2)

        # 画曲面（evalpts）
        if self.display_evalpts:
            for index, evalpt in enumerate(self.evalpts):
                # evalpt 是二维网格点 (nu, nv, 3)
                surf_points = self.ensure_3d(evalpt.reshape(-1, evalpt.shape[-1]))

                # 网格行列数
                nu, nv = evalpt.shape[:2]

                # 构造拓扑结构（每个cell是四边形）
                # PyVista Quad单元需要四个点索引，排成矩形网格
                faces = []
                for i in range(nu - 1):
                    for j in range(nv - 1):
                        # Quad由4个点组成，格式是 [4, idx0, idx1, idx2, idx3]
                        idx0 = i * nv + j
                        idx1 = idx0 + 1
                        idx2 = idx0 + nv + 1
                        idx3 = idx0 + nv
                        faces.extend([4, idx0, idx1, idx2, idx3])

                faces = np.array(faces)

                mesh = pv.PolyData(surf_points, faces)
                plotter.add_mesh(mesh, scalars=surf_points[:, 2], cmap=surf_cmaps, opacity=0.85, label=f"NURBS Surface {index}")

        # 画采样点
        if self.data_point is not None:
            surface_point = self.ensure_3d(np.asarray(self.data_point))
            if surface_point.shape[-1] != 3:
                raise ValueError(f"Surface points should be in R^3")
            cloud = pv.PolyData(surface_point)
            plotter.add_mesh(cloud, color='green', point_size=10, render_points_as_spheres=True, label="Sampling Points")

        if self.bounding_volume:
            for slab in self.bounding_volume:
                self.create_triangle_slab(plotter, slab['point1'], slab['point2'], slab['point3'], slab['radius'])

        # 添加图例（PyVista里默认没有图例按钮，这里用文本注释）
        if self.display_legend:
            labels = []
            if self.display_ctrlpts:
                labels.append(("Control Points", "red"))
            if self.display_evalpts:
                labels.append(("NURBS Surface", "black"))
            if self.data_point is not None:
                labels.append(("Sampling Points", "green"))

            for i, (text, color) in enumerate(labels):
                plotter.add_text(text, position=[10, 30 + i * 20], font_size=12, color=color)

        # 设置坐标轴显示
        if self.display_axes:
            plotter.show_axes()
        else:
            plotter.hide_axes()

        # 设置坐标轴标签
        if self.display_labels:
            plotter.show_axes()
            plotter.show_grid()

        # 显示
        plotter.show()

        if self.export_file:
            from third_party.pyevtk.hl import gridToVTK
            interpolatedPointsU = np.ascontiguousarray(self.evalpts[:, :, 0])
            interpolatedPointsV = np.ascontiguousarray(self.evalpts[:, :, 1])
            interpolatedPointsW = np.ascontiguousarray(self.evalpts[:, :, 2])
            gridToVTK(f"NurbsSurface", interpolatedPointsU, interpolatedPointsV, interpolatedPointsW, cellData={}, pointData={})

    def PlotVolume(self):
        """
        用 pyvista 绘制三维体积控制点和评估点（点云形式），带坐标轴刻度和图例
        """
        plotter = pv.Plotter()

        legend_entries = []

        # 绘制控制点
        if self.display_ctrlpts:
            ctrl_pts = self.ctrlpts  # 形状 (N, 3)
            ctrlpt_cloud = pv.PolyData(ctrl_pts)
            plotter.add_mesh(ctrlpt_cloud, color='red', point_size=10, render_points_as_spheres=True)
            legend_entries.append(('Control points', 'red'))

        # 绘制评估点体积
        if self.display_evalpts:
            eval_pts = self.evalpts  # 形状 (M, 3)
            evalpt_cloud = pv.PolyData(eval_pts)
            plotter.add_mesh(evalpt_cloud, color='black', point_size=6, render_points_as_spheres=True, opacity=self.alpha)
            legend_entries.append(('NURBS volume', 'black'))

        # 坐标轴显示
        plotter.show_axes()

        # 添加图例
        if self.display_legend:
            for label, color in legend_entries:
                plotter.add_legend_entry(label, color)
            plotter.add_legend()

        # 显示坐标轴刻度
        plotter.show_grid()

        # 显示
        plotter.show()

    def create_triangle_slab(self, plotter, p1, p2, p3, rad):
        triangle = pv.PolyData(np.array([p1, p2, p3]))
        triangle.faces = np.hstack([[3,0,1,2]])
        triangle.compute_normals(inplace=True)
        slab1 = triangle.extrude((0,0,rad), capping=True)
        slab2 = triangle.extrude((0,0,-rad), capping=True)
        plotter.add_mesh(triangle, color="red", opacity=0.3, show_edges=True)
        plotter.add_mesh(slab1, color="lightblue", opacity=0.7, show_edges=True)
        plotter.add_mesh(slab2, color="lightblue", opacity=0.7, show_edges=True)
                