
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt

from src.nurbs.Utilities import point_to_edges, point_to_point

 
def closest_centroid(points, centroids, norm="L2"):
    distances = point_to_point(points[:, np.newaxis], centroids, axis=2, norm=norm)
    return np.argmin(distances, axis=1)

def closest_edge(points, edges, norm="L2"):
    num_edges = edges.shape[0]
    distance = np.zeros((points.shape[0], num_edges))
    for idp, point in enumerate(points):
        for e in range(num_edges):
            distance[idp, e] = point_to_edges(point, edges[e][0], edges[e][1], norm)
    return np.argmin(distance, axis=1)
 
def update_centroids(points, labels, ngroup):
    return np.array([points[labels == i].mean(axis=0) for i in range(ngroup)])

def fit_line_L1norm(points, max_iter=100, tol=1e-6, delta_init=1e-3, epsilon=1e-8):
    x = points[:, 0]
    y = points[:, 1]
    n = len(x)
    
    S_x  = np.sum(x)
    S_y  = np.sum(y)
    S_xx = np.sum(x*x)
    S_xy = np.sum(x*y)
    D0   = S_xx * n - S_x**2
    a = (S_xy*n - S_x*S_y) / D0
    b = np.median(y - a*x)
    
    delta = delta_init
    for _ in range(max_iter):
        b = np.median(y - a*x)
        loss_current = ((abs(a) + 1) / (a**2 + 1)) * np.sum(np.abs(y - (a*x + b)))
        
        a_plus = a + delta
        b_plus = np.median(y - a_plus*x)
        loss_plus = ((abs(a_plus) + 1) / (a_plus**2 + 1)) * np.sum(np.abs(y - (a_plus*x + b_plus)))
        a_minus = a - delta
        b_minus = np.median(y - a_minus*x)
        loss_minus = ((abs(a_minus) + 1) / (a_minus**2 + 1)) * np.sum(np.abs(y - (a_minus*x + b_minus)))
        
        if loss_plus < loss_current and loss_plus <= loss_minus:
            a_new, b_new, loss_new = a_plus, b_plus, loss_plus
        elif loss_minus < loss_current and loss_minus < loss_plus:
            a_new, b_new, loss_new = a_minus, b_minus, loss_minus
        else:
            a_new, b_new, loss_new = a, b, loss_current
        if loss_new < loss_current:
            a, b = a_new, b_new
        else:
            delta *= 0.5
        if abs(loss_current - loss_new) < tol:
            break
    return a, b

def fit_line_L2norm(points):
    centroid = np.mean(points, axis=0)
    X = points - centroid
    U, S, Vt = np.linalg.svd(X)
    n = Vt[1] 
    c = -np.dot(n, centroid)
    if np.abs(n[1]) > 1e-8:
        a = -n[0] / n[1]
        b = -c / n[1]
    else:
        a = np.inf
        b = centroid[0]
    return a, b

def update_edges(points, labels, ngroup):
    edges = np.zeros((ngroup, 2, points.shape[1]))
    for i in range(ngroup):
        point_cloud = points[labels == i]
        a, b = fit_line_L1norm(point_cloud)
        v = np.array([1.0, a])
        v = v / np.linalg.norm(v)
        c = np.array([np.mean(point_cloud[:, 0]), np.mean(point_cloud[:, 1])])
        t = np.dot(point_cloud - c, v)
        t_min = t.min()
        t_max = t.max()
        p1 = c + t_min * v
        p2 = c + t_max * v
        edges[i][0] = p1
        edges[i][1] = p2
    return edges

def centroid_error(new_centroids, centroids):
    return np.sum(np.linalg.norm(np.abs(new_centroids - centroids)))

def edge_error(new_edges, edges):
    dist = 0.
    for i in range(new_edges.shape[0]):
        dist += max(directed_hausdorff(new_edges[i], edges[i])[0], directed_hausdorff(edges[i], new_edges[i])[0])
    return dist

def visualization(ngroup, labels, point_clouds, centroids=None, edges=None):
    fig = plt.figure()
    if point_clouds.shape[1] == 2:
        for i in range(ngroup):
            plt.scatter(point_clouds[labels == i, 0], point_clouds[labels == i, 1], alpha=0.6, edgecolors='w', s=50, label=f'cluster{i}')
            if edges is not None:
                plt.plot(edges[i,:,0], edges[i,:,1])
        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='X', label='Centroids')
    elif point_clouds.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i in range(ngroup):
            ax.scatter(point_clouds[labels == i, 0], point_clouds[labels == i, 1], point_clouds[labels == i, 2], alpha=0.6, edgecolors='w', s=50, label=f'cluster{i}')
            if edges is not None:
                plt.plot(edges[i,:,0], edges[i,:,1], edges[i,:,2])
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=50, marker='X', label='Centroids')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
 
def kmean(point_clouds, centroids=None, edges=None, norm="L2", max_iters=100, tol=1e-4, visualize=False):
    point_clouds = np.asarray(point_clouds)

    find_closest, update_characteristic, calculate_error, targets = None, None, None, None
    if centroids is not None:  
        centroids = np.asarray(centroids)
        targets = centroids.copy()
        find_closest = closest_centroid
        update_characteristic = update_centroids
        calculate_error = centroid_error
    elif edges is not None:
        edges = np.asarray(edges)
        targets = edges.copy()
        find_closest = closest_edge
        update_characteristic = update_edges
        calculate_error = edge_error
    ngroup = targets.shape[0]

    for _ in range(max_iters):
        labels = find_closest(point_clouds, targets, norm)
        new_targets = update_characteristic(point_clouds, labels, ngroup)
        if calculate_error(new_targets, targets) < tol:
            break
        targets = new_targets

    if visualize:
        visualization(ngroup, labels, point_clouds, centroids, edges)
    return labels, targets