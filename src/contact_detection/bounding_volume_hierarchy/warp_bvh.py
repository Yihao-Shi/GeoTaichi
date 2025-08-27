import warp as wp

from src.contact_detection.bounding_volume_hierarchy.AABB import AABB


class BVH:
    def __init__(self, primitive_count):
        self.aabb = AABB(primitive_count)
        self.tri_bvh = wp.Bvh()
        
        self.device = wp.get_device()
        self.lower_bounds_tris = wp.array(shape=(primitive_count,), dtype=wp.vec3, device=self.device)
        self.upper_bounds_tris = wp.array(shape=(primitive_count,), dtype=wp.vec3, device=self.device)

    def build(self):
        self.aabb.update_patch_aabb
        wp.bvh_query_next