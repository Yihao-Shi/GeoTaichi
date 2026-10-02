import numpy as np
import pyvista as pv

from itertools import product


def segment_segment_dist(p1, q1, p2, q2):
    """
    Compute the shortest distance between two line segments (p1,q1) and (p2,q2).
    Returns the distance and the closest points on each segment.
    """
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    denom = a * c - b * b
    s = 0.0
    t = 0.0

    if denom != 0:
        s = (b * e - c * d) / denom
        s = np.clip(s, 0, 1)
    else:
        s = 0.0

    t = (b * s + e) / c if c != 0 else 0.0
    t = np.clip(t, 0, 1)

    # Recalculate s if t is out of bounds
    if t < 0:
        t = 0
        s = np.clip(-d / a if a != 0 else 0, 0, 1)
    elif t > 1:
        t = 1
        s = np.clip((b - d) / a if a != 0 else 0, 0, 1)

    closest_point_line1 = p1 + s * u
    closest_point_line2 = p2 + t * v
    dist = np.linalg.norm(closest_point_line1 - closest_point_line2)
    return dist, closest_point_line1, closest_point_line2

def point_to_quad_dist(p, quad):
    """
    Compute the distance from point p to a quadrilateral defined by four vertices (quad).
    Projects point onto the quad's plane and checks if projection lies inside the quad.
    If inside, returns perpendicular distance to plane.
    Otherwise, returns shortest distance to the quad edges.
    """
    v0, v1, v2, v3 = quad
    normal = np.cross(v1 - v0, v3 - v0)
    normal /= np.linalg.norm(normal)

    vec = p - v0
    dist_to_plane = np.dot(vec, normal)
    proj = p - dist_to_plane * normal

    def point_in_tri(pnt, a, b, c):
        v0 = c - a
        v1 = b - a
        v2 = pnt - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = dot00 * dot11 - dot01 * dot01
        if denom == 0:
            return False
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom
        return (u >= 0) and (v >= 0) and (u + v <= 1)

    def point_in_quad(pt, quad):
        return point_in_tri(pt, quad[0], quad[1], quad[2]) or point_in_tri(pt, quad[0], quad[2], quad[3])

    if point_in_quad(proj, quad):
        return abs(dist_to_plane), proj
    else:
        edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v0)]
        min_dist = float('inf')
        closest_point = None
        for pa, pb in edges:
            d, cp = segment_point_dist(pa, pb, p)
            if d < min_dist:
                min_dist = d
                closest_point = cp
        return min_dist, closest_point

def segment_point_dist(p1, p2, p):
    """
    Compute distance from point p to line segment p1-p2.
    Returns the distance and the closest point on the segment.
    """
    line_vec = p2 - p1
    p_vec = p - p1
    t = np.dot(p_vec, line_vec) / np.dot(line_vec, line_vec)
    t = np.clip(t, 0, 1)
    closest = p1 + t * line_vec
    dist = np.linalg.norm(closest - p)
    return dist, closest

def point_triangle_distance(p, tri):
    """
    Compute the shortest distance between point p and triangle tri (3 points).
    Return distance and closest point on triangle.
    
    Parameters:
    - p: np.ndarray (3,)
    - tri: np.ndarray (3,3)
    
    Returns:
    - dist: float
    - closest_point: np.ndarray (3,)
    """
    a, b, c = np.asarray(tri)
    ab = b - a
    ac = c - a
    ap = p - a

    # Compute barycentric coordinates
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)

    if d1 <= 0 and d2 <= 0:
        return np.linalg.norm(p - a), (1.0, 0.0, 0.0)  # Closest to vertex a

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return np.linalg.norm(p - b), (0.0, 1.0, 0.0)  # Closest to vertex b

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return np.linalg.norm(p - c), (0.0, 0.0, 1.0)  # Closest to vertex c

    # Check edge regions
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        point = a + v * ab
        return np.linalg.norm(p - point), (1 - v, v, 0.0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        point = a + w * ac
        return np.linalg.norm(p - point), (1 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        point = b + w * (c - b)
        return np.linalg.norm(p - point), (0.0, 1 - w, w)

    # Inside face region
    denom = 1.0 / (np.dot(ab, ab) * np.dot(ac, ac) - np.dot(ab, ac)**2)
    v = (np.dot(ac, ac) * np.dot(ap, ab) - np.dot(ab, ac) * np.dot(ap, ac)) * denom
    w = (np.dot(ab, ab) * np.dot(ap, ac) - np.dot(ab, ac) * np.dot(ap, ab)) * denom
    u = 1 - v - w
    point = u * a + v * b + w * c
    return np.linalg.norm(p - point), (u, v, w)

def capsule_capsule_distance(line1, radius1, line2, radius2):
    d, _, _ = segment_segment_dist(line1[0], line1[1], line2[0], line2[1]) 
    return d - radius1 - radius2

def slab_slab_distance(quad1, radius1, quad2, radius2):
    """
    Compute distance between two slabs, each defined by a quadrilateral (quad) and a radius (half thickness).
    Returns the minimal distance minus the sum of radii.
    """
    distances = []

    # Edges of both quads
    edges1 = [(quad1[i], quad1[(i+1)%4]) for i in range(4)]
    edges2 = [(quad2[i], quad2[(i+1)%4]) for i in range(4)]

    # Segment-segment distances between edges of both quads
    for e1 in edges1:
        for e2 in edges2:
            d, _, _ = segment_segment_dist(np.array(e1[0]), np.array(e1[1]),
                                          np.array(e2[0]), np.array(e2[1]))
            distances.append(d)

    # Quad1 vertices to Quad2 surface distances
    for p in quad1:
        d, _ = point_to_quad_dist(np.array(p), np.array(quad2))
        distances.append(d)

    # Quad2 vertices to Quad1 surface distances
    for p in quad2:
        d, _ = point_to_quad_dist(np.array(p), np.array(quad1))
        distances.append(d)

    min_dist = min(distances)

    # Final slab distance subtracting radii sum
    slab_dist = min_dist - (radius1 + radius2)
    return slab_dist

def slab_to_plane_distance_z0(quad, radius, z0):
    """
    Compute the distance between a slab (defined by quad + radius) and a horizontal plane z=z0.
    Return the minimal distance after subtracting slab radius.
    
    Parameters:
    - quad: list or array of 4 points, each shape (3,)
    - radius: float, slab half thickness
    - z0: float, z-plane
    
    Returns:
    - distance: float, distance from slab surface to z=z0 (negative means penetration)
    """
    quad = np.array(quad)  # shape (4,3)
    z_coords = quad[:, 2]
    
    # Distance from each vertex to z=z0
    dists = np.abs(z_coords - z0)
    min_vertex_dist = np.min(dists)
    
    # Compute minimal distance from slab edges to z=z0 plane
    def segment_plane_distance(p1, p2, z0):
        z1, z2 = p1[2], p2[2]
        if (z1 - z0) * (z2 - z0) <= 0:
            # Edge crosses the plane => minimal distance is zero
            return 0.0
        else:
            # Otherwise, closest endpoint
            return min(abs(z1 - z0), abs(z2 - z0))
    
    edge_pairs = [(quad[i], quad[(i+1)%4]) for i in range(4)]
    min_edge_dist = min(segment_plane_distance(p1, p2, z0) for p1, p2 in edge_pairs)
    
    # The closest distance from quad to plane (without considering slab radius)
    min_dist = min(min_vertex_dist, min_edge_dist)
    
    # Final slab-to-plane distance (subtracting slab radius)
    return min_dist - radius

def triangle_slab_plane_distance(triangle, radius, z0):
    """
    Compute slab (triangle + radius) to z=z0 plane distance.
    """
    triangle = np.asarray(triangle).reshape(3, 3)
    z_coords = triangle[:, 2]
    min_dist = np.min(np.abs(z_coords - z0))
    return min_dist - radius

def triangle_slab_triangle_slab_distance(tri1, radius1, tri2, radius2):
    """
    Compute slab-to-slab distance between two triangles.
    """
    tri1 = np.asarray(tri1).reshape(3, 3)
    tri2 = np.asarray(tri2).reshape(3, 3)
    
    d0 = triangle_triangle_distance(tri1, tri2)
    return d0 - (radius1 + radius2)

def triangle_triangle_distance(tri1, tri2, return_weights=False):
    """
    Compute shortest distance between two triangles in 3D.
    """
    if return_weights:
        min_dist = np.inf
        closest_info = []
        # vertex to triangle
        for p in tri1:
            distance, weights = point_triangle_distance(p, tri2)
            if distance < min_dist:
                min_dist = distance
                wA = compute_barycentric_coords(tri1, p)
                wB = weights
                closest_info = [distance, wA, wB]
        for p in tri2:
            distance, weights = point_triangle_distance(p, tri1)
            if distance < min_dist:
                min_dist = distance
                wB = compute_barycentric_coords(tri2, p)
                wA = weights
                closest_info = [distance, wA, wB]
        # edge to edge
        edges1 = [(tri1[i], tri1[(i+1)%3]) for i in range(3)]
        edges2 = [(tri2[i], tri2[(i+1)%3]) for i in range(3)]
        for (p1, q1), (p2, q2) in product(edges1, edges2):
            distance, cp1, cp2 = segment_segment_dist(p1, q1, p2, q2)
            if distance < min_dist:
                min_dist = distance
                wA = compute_barycentric_coords(tri1, cp1)
                wB = compute_barycentric_coords(tri2, cp2)
                closest_info = [distance, wA, wB]
        return closest_info
    else:
        dists = []
        # vertex to triangle
        for p in tri1:
            dists.append(point_triangle_distance(p, tri2)[0])
        for p in tri2:
            dists.append(point_triangle_distance(p, tri1)[0])
        # edge to edge
        edges1 = [(tri1[i], tri1[(i+1)%3]) for i in range(3)]
        edges2 = [(tri2[i], tri2[(i+1)%3]) for i in range(3)]
        for (p1, q1), (p2, q2) in product(edges1, edges2):
            dists.append(segment_segment_dist(p1, q1, p2, q2)[0])
        return min(dists)

def compute_barycentric_coords(tri, p):
    a, b, c = tri
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return [1.0, 0.0, 0.0]  # degenerate, fallback
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return [u, v, w]

def visualize(tri1, tri2):
    dist, wA, wB = triangle_triangle_distance(tri1, tri2, return_weights=True)

    p1 = wA[0] * tri1[0] + wA[1] * tri1[1] + wA[2] * tri1[2]
    p2 = wB[0] * tri2[0] + wB[1] * tri2[1] + wB[2] * tri2[2]

    # 可视化
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.show_grid()
    plotter.set_background("white")

    # 添加两个三角形
    mesh1 = pv.PolyData(np.asarray(tri1), faces=[3, 0, 1, 2])
    mesh2 = pv.PolyData(np.asarray(tri2), faces=[3, 0, 1, 2])
    plotter.add_mesh(mesh1, color='red', opacity=0.5, label="Triangle 1")
    plotter.add_mesh(mesh2, color='blue', opacity=0.5, label="Triangle 2")

    # 添加最近点和线段
    plotter.add_mesh(pv.Line(p1, p2), color='black', line_width=3)
    plotter.add_mesh(pv.Sphere(radius=0.01, center=p1), color='red')
    plotter.add_mesh(pv.Sphere(radius=0.01, center=p2), color='blue')

    # 距离标注（可选）
    midpoint = (p1 + p2) / 2
    plotter.add_point_labels([midpoint], [f"{dist:.4f}"], font_size=14, text_color="black")

    plotter.show()

def display2D(tri1, tri2):
    import pygame
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Triangle-Triangle Distance")

    font = pygame.font.SysFont("Arial", 24)

    # 2D projection helpers
    def to_screen(p):
        return int(p[0] + WIDTH // 2), int(HEIGHT // 2 - p[1])

    dragging = None
    clock = pygame.time.Clock()

    def draw_triangle(tri, color):
        pts = [to_screen(p) for p in tri]
        pygame.draw.polygon(screen, color, pts, 2)

    running = True
    while running:
        screen.fill((255, 255, 255))
        draw_triangle(tri1, (255, 0, 0))
        draw_triangle(tri2, (0, 0, 255))

        closest_info = triangle_triangle_distance(tri1, tri2, return_weights=True)
        dist = closest_info[0]
        wA = closest_info[1]
        wB = closest_info[2]
        p1 = wA[0] * tri1[0] + wA[1] * tri1[1] + wA[2] * tri1[2]
        p2 = wB[0] * tri2[0] + wB[1] * tri2[1] + wB[2] * tri2[2]
        pygame.draw.circle(screen, (0, 0, 255), to_screen(p1), 5)
        pygame.draw.circle(screen, (255, 0, 0), to_screen(p2), 5)
        pygame.draw.line(screen, (0, 0, 0), to_screen(p1), to_screen(p2), 1)

        dist_text = font.render(f"Distance: {dist:.2f}", True, (0, 0, 0))
        screen.blit(dist_text, (20, 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                mpos = np.array([mx - WIDTH // 2, HEIGHT // 2 - my])
                for i, tri in enumerate([tri1, tri2]):
                    for j, p in enumerate(tri):
                        if np.linalg.norm(mpos - p) < 10:
                            dragging = (i, j)
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                i, j = dragging
                mx, my = pygame.mouse.get_pos()
                mpos = np.array([mx - WIDTH // 2, HEIGHT // 2 - my])
                if i == 0:
                    tri1[j] = mpos
                else:
                    tri2[j] = mpos

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()