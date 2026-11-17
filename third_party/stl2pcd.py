import sys, os
import numpy as np
from stl import mesh
import open3d as o3d
from collections import defaultdict

def stl_to_pcd(stl_file, pcd_file, boundary_file, interior_file):
    try:
        stl_mesh = mesh.Mesh.from_file(stl_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"STL file not found: {stl_file}")
        
    # Load STL file
    stl_mesh = mesh.Mesh.from_file(stl_file)
    
    # Extract vertices from STL file
    unique_vertices, indices = np.unique(stl_mesh.vectors.reshape(-1, 3), axis=0, return_inverse=True)
    
    # Create Open3D point cloud
    edge_count = defaultdict(int)
    for triangle in indices.reshape(-1, 3):
        edges = [(triangle[i], triangle[j]) for i, j in [(0, 1), (1, 2), (2, 0)]]
        for edge in edges:
            edge_count[tuple(sorted(edge))] += 1
            
    boundary_points = set()
    for edge, count in edge_count.items():
        if count == 1:  # Boundary edges occur only in one triangle
            boundary_points.update(edge)
    
    # Separate boundary and interior points
    boundary_vertices = unique_vertices[list(boundary_points)]
    interior_vertices = np.array([v for i, v in enumerate(unique_vertices) if i not in boundary_points])
    
    # Save all points as PCD
    all_points_cloud = o3d.geometry.PointCloud()
    all_points_cloud.points = o3d.utility.Vector3dVector(unique_vertices)
    o3d.io.write_point_cloud(pcd_file, all_points_cloud)
    
    # Save boundary points as PCD
    boundary_cloud = o3d.geometry.PointCloud()
    boundary_cloud.points = o3d.utility.Vector3dVector(boundary_vertices)
    o3d.io.write_point_cloud(boundary_file, boundary_cloud)
    
    # Save interior points as PCD
    interior_cloud = o3d.geometry.PointCloud()
    interior_cloud.points = o3d.utility.Vector3dVector(interior_vertices)
    o3d.io.write_point_cloud(interior_file, interior_cloud)
    
    print(f"Saved all points to {pcd_file}, boundary points to {boundary_file}, and interior points to {interior_file}.")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise RuntimeError("Usage: python stl_to_pcd.py /path/to/input_file /path/to/output_file /path/to/boundary_file /path/to/interior_file")
    input_file = sys.argv[1] + ".stl"
    output_file = sys.argv[2] + ".pcd"
    boundary_output_file = sys.argv[3] + ".pcd"
    interior_output_file = sys.argv[4] + ".pcd"
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    stl_to_pcd(input_file, output_file, boundary_output_file, interior_output_file)
