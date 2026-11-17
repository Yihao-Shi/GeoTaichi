import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_nd_slice(f, ranges, fixed_values={}, dim_x=0, dim_y=1, mode="contourf", resolution=100, mark_points=None):
    """
    f: callable, function f(x0, x1, ..., xn)
    ranges: list of (min, max), the range for each dimension
    fixed_values: dict, {dim_index: value} fixed dimensions
    dim_x, dim_y: int, which two dimensions to plot on x and y axes
    mode: 'contourf' or 'surface' for 2D contour or 3D surface plot
    resolution: int, number of points per axis for grid
    mark_points: list of tuples [(x0, x1, ..., xn), ...], points to mark on the plot (optional)
    """
    total_dim = len(ranges)
    x_range = np.linspace(ranges[dim_x][0], ranges[dim_x][1], resolution)
    y_range = np.linspace(ranges[dim_y][0], ranges[dim_y][1], resolution)
    X, Y = np.meshgrid(x_range, y_range)

    # Construct inputs for all dimensions
    inputs = []
    for i in range(total_dim):
        if i == dim_x:
            inputs.append(X)
        elif i == dim_y:
            inputs.append(Y)
        elif i in fixed_values:
            inputs.append(np.full_like(X, fixed_values[i]))
        else:
            raise ValueError(f"Dimension {i} is not specified in fixed_values or plotting dims")

    # Evaluate function
    Z = np.empty_like(X)
    for idx in np.ndindex(X.shape):
        args = []
        for dim in range(total_dim):
            args.append(inputs[dim][idx])
        Z[idx] = f(*args)

    plt.figure(figsize=(7, 6))
    if mode == "contourf":
        cp = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
        plt.colorbar(cp)
        plt.xlabel(f"Dimension {dim_x}")
        plt.ylabel(f"Dimension {dim_y}")
        plt.title("Contourf Slice")

        # Mark points if given
        if mark_points is not None:
            for pt in mark_points:
                # Extract x,y coords from pt tuple
                x_pt, y_pt = pt[dim_x], pt[dim_y]
                # Compute function value at pt
                z_pt = f(*pt)
                plt.plot(x_pt, y_pt, "ro")
                plt.text(x_pt, y_pt, f"{z_pt:.3f}", color="white", fontsize=9)

    elif mode == "surface":
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
        ax.set_xlabel(f"Dimension {dim_x}")
        ax.set_ylabel(f"Dimension {dim_y}")
        ax.set_zlabel("f(x)")
        plt.title("Surface Slice")

        # Mark points if given
        if mark_points is not None:
            for pt in mark_points:
                x_pt, y_pt = pt[dim_x], pt[dim_y]
                z_pt = f(*pt)
                ax.scatter(x_pt, y_pt, z_pt, color="r", s=50)
                ax.text(x_pt, y_pt, z_pt, f"{z_pt:.3f}", color="black")

    else:
        raise ValueError("mode must be 'contourf' or 'surface'")

    plt.show()
