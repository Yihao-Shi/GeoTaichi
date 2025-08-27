import taichi as ti

from src.contact_detection.bounding_volume_hierarchy.LBVH import LBvh
from src.utils.constants import DBL_MAX


@ti.data_oriented
class RayTracingCollision:
    def __init__(self, primitive_count):
        self.primitive_count = primitive_count
        self.lbvh = LBvh(primitive_count)

    @ti.func
    def ray_trace_brute_force(self, origin, direction):
        hit_t       = DBL_MAX
        hit_pos     = ti.Vector([0.0, 0.0, 0.0]) 
        hit_bary    = ti.Vector([0.0, 0.0]) 
        hit_prim    = -1

        prim_index = 0
        while prim_index < self.primitive_count:
            t, pos, bary = self.intersect_prim(origin, direction, prim_index)
            if t < hit_t and t > 0.0:
                hit_t = t
                hit_pos = pos
                hit_bary = bary
                hit_prim = prim_index
            prim_index +=1
        return hit_t, hit_pos, hit_bary, hit_prim

    @ti.func
    def signed_distance_brute_force(self, p, water_tight=1):
        sd = DBL_MAX
        prim_index = 0
        while prim_index < self.primitive_count:
            t = self.unsigned_distance_triangle(prim_index, p)
            if (t < sd) :
                sd = t
            prim_index +=1
        return sd, ti.Vector([0.0, 1.0, 0.0])

    ############Client Interface############## 
    @ti.kernel
    def ray_trace_cpu(self,  origin: ti.types.vector(3, float), direction:ti.types.vector(3, float))-> ti.types.vector(4, float):
        direction_n = ti.math.normalize(direction)
        hit_t, hit_pos, hit_bary, hit_prim = self.ray_trace(origin,direction_n)
        return ti.Vector([hit_t, hit_pos[0], hit_pos[1], hit_pos[2]])
    
    @ti.kernel
    def singed_distance_cpu(self,  point: ti.types.vector(3, float))-> ti.types.vector(7, float):
        t, closest = self.signed_distance(point)
        sign = 1.0
        if (t > 0.0):
            sign = -1.0
        gradient  = (closest - point) * sign
        gradient = gradient.normalized()
        return ti.Vector([t, closest[0], closest[1], closest[2], sign * gradient[0], sign * gradient[1], sign * gradient[2]])

    ############Interface##############
    @ti.func
    def ray_trace(self, origin, direction):
        hit_t       = DBL_MAX
        hit_pos     = ti.Vector([0.0, 0.0, 0.0]) 
        hit_bary    = ti.Vector([0.0, 0.0]) 
        hit_prim    = -1
        MAX_SIZE    = 32
        stack       = ti.Vector.zero(float, MAX_SIZE)
        stack_pos   = 0

        # depth first use stack
        while (stack_pos >= 0) & (stack_pos < MAX_SIZE):
            #pop
            node_index = stack[stack_pos]
            stack[stack_pos] = 0
            stack_pos  = stack_pos - 1
            leaf_index = self.lbvh.get_node_leaf_index(node_index)

            if leaf_index >= 0 :
                leaf_count = self.lbvh.get_node_leaf_count(node_index)
                count = 0
                while count < leaf_count:
                    prim_index = self.lbvh.get_node_prim_index(leaf_index, count) 
                    t, pos, bary = self.intersect_triangle(origin, direction, prim_index)
                    if ( t < hit_t ) & (t > 0.0):
                        hit_t       = t
                        hit_pos     = pos
                        hit_bary    = bary
                        hit_prim    = prim_index
                    count+=1
            else:
                # seems more clever way  
                left_node, right_node = self.lbvh.get_node_child(node_index)
                lmin_v, lmax_v = self.lbvh.get_node_min_max( left_node)
                rmin_v, rmax_v = self.lbvh.get_node_min_max( right_node)

                retl, minl, maxl = self.slabs(origin, direction,lmin_v,lmax_v)
                retr, minr, maxr = self.slabs(origin, direction,rmin_v,rmax_v)
                #ckeck_count += 2
                if minr < hit_t  and minr<=minl :
                    if minl < hit_t :
                        stack_pos += 1
                        stack[stack_pos] = left_node
                    stack_pos += 1
                    stack[stack_pos] = right_node

                if minl < hit_t and minl<minr :
                    if minr < hit_t :
                        stack_pos += 1
                        stack[stack_pos] = right_node
                    stack_pos += 1
                    stack[stack_pos] = left_node
        assert stack_pos >= MAX_SIZE, "overflow, need larger stack"
        return  hit_t, hit_pos, hit_bary, hit_prim

    @ti.func
    def slabs(self, origin, direction, minv, maxv):
        # most effcient algrithm for ray intersect aabb 
        # en vesrion: https:#www.researchgate.net/publication/220494140_An_Efficient_and_Robust_Ray-Box_Intersection_Algorithm
        ret  = 1
        tmin = 0.0
        tmax = DBL_MAX
        for i in ti.static(range(3)):
            if abs(direction[i]) < 0.000001:
                if ( (origin[i] < minv[i]) | (origin[i] > maxv[i])):
                    ret = 0
                    tmin = DBL_MAX
            else:
                ood = 1.0 / direction[i] 
                t1 = (minv[i] - origin[i]) * ood 
                t2 = (maxv[i] - origin[i]) * ood
                if(t1 > t2):
                    temp = t1 
                    t1 = t2
                    t2 = temp 
                if(t1 > tmin):
                    tmin = t1
                if(t2 < tmax):
                    tmax = t2 
                if(tmin > tmax) :
                    ret=0
                    tmin = DBL_MAX
        return ret, tmin, tmax

    @ti.func
    def signed_distance(self,  p, water_tight=1):
        closest_prim= -1
        closest_p   = ti.Vector([0.0,0.0,0.0])
        sd = DBL_MAX
        MAX_SIZE    = 32
        stack       = ti.Vector([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stack_pos   = 0

        # depth first use stack
        while (stack_pos >= 0) & (stack_pos < MAX_SIZE):
            #pop
            node_index = stack[stack_pos]
            stack[stack_pos] = 0
            stack_pos  = stack_pos-1
            leaf_index = self.lbvh.get_node_leaf_index(node_index)

            if leaf_index > 0 :
                leaf_count = self.lbvh.get_node_leaf_count(node_index)
                count = 0
                while count < leaf_count:
                    prim_index = self.lbvh.get_node_prim_index(leaf_index, count)
                    t,cp = self.unsigned_distance(prim_index, p)
                    if  t < sd :
                        sd       = t
                        closest_prim = prim_index
                        closest_p    = cp
                    count+=1
            else:
                #depth first search
                min_v,max_v = self.lbvh.get_node_min_max( node_index)
                t,closet    = self.sdf_box(p,min_v,max_v)
                if t < sd :
                    #push
                    left_node,right_node = self.lbvh.get_node_child(node_index)
                    #depth first search
                    stack[stack_pos+1] = left_node
                    stack[stack_pos+2] = right_node
                    stack_pos              += 2

        assert stack_pos >= MAX_SIZE, "overflow, need larger stack"

        #in out check
        sd_extra    = 0.0
        min0,max0 = self.lbvh.get_node_min_max(0)

        if self.inside_box(p, min0,max0) == 1:
            if (water_tight):
                if (self.is_inside(closest_prim, p) ):
                    sd = -sd
            else:
                #   This is not a watertight mesh
                #   We need to sample more direction
                origin = p
                direction = ti.math.normalize((p - closest_p))
                hit_t, hit_pos, hit_bary,hit_prim = self.ray_trace(origin, direction)
                if (hit_t < DBL_MAX):
                    sd = -sd
        return sd + sd_extra, closest_p

    @ti.func
    def unsigned_distance_triangle(self, index, p):
        dis     = DBL_MAX
        closet  = p

        #https:#www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
        p0 = self.tri_p1(index)
        p1 = self.tri_p2(index)
        p2 = self.tri_p3(index)

        diff  = p0 - p
        edge0 = p1 - p0
        edge1 = p2 - p0

        a00 = edge0.dot(edge0)
        a01 = edge0.dot(edge1)
        a11 = edge1.dot(edge1)
        b0 = diff.dot(edge0)
        b1 = diff.dot(edge1)
        det = ti.math.max(a00 * a11 - a01 * a01, 0.0)
        s = a01 * b1 - a11 * b0
        t = a01 * b0 - a00 * b1

        if (s + t <= det):
            if (s < 0.0):
                if (t < 0.0):  
                    if (b0 < 0.0):
                        t = 0.0
                        if (-b0 >= a00):
                            s = 1.0
                        else:
                            s = -b0 / a00
                    else:
                        s = 0.0
                        if (b1 >= 0.0):
                            t = 0.0
                        elif (-b1 >= a11):
                            t = 1.0
                        else:
                            t = -b1 / a11
                else:  # region 3
                    s = 0.0
                    if (b1 >= 0.0):
                        t = 0.0
                    elif (-b1 >= a11):
                        t = 1.0
                    else:
                        t = -b1 / a11
            elif (t < 0.0) : # region 5
                t = 0.0
                if (b0 >= 0.0):
                    s = 0.0
                elif (-b0 >= a00):
                    s = 1.0
                else:
                    s = -b0 / a00
            else : # region 0
                # minimum at interior point
                s /= det
                t /= det
        else:
            tmp0 = 0.0
            tmp1  = 0.0
            numer  = 0.0
            denom = 0.0

            if (s < 0.0):  # region 2
                tmp0 = a01 + b0
                tmp1 = a11 + b1
                if (tmp1 > tmp0):
                    numer = tmp1 - tmp0
                    denom = a00 - 2.0 * a01 + a11
                    if (numer >= denom):
                        s = 1.0
                        t = 0.0
                    else:
                        s = numer / denom
                        t = 1.0 - s
                else:
                    s = 0.0
                    if (tmp1 <= 0.0):
                        t = 1.0
                    elif (b1 >= 0.0):
                        t = 0.0
                    else:
                        t = -b1 / a11
            elif (t < 0.0) : # region 6
                tmp0 = a01 + b1
                tmp1 = a00 + b0
                if (tmp1 > tmp0):
                    numer = tmp1 - tmp0
                    denom = a00 - 2.0 * a01 + a11
                    if (numer >= denom):
                        t = 1.0
                        s = 0.0
                    else:
                        t = numer / denom
                        s = 1.0 - t
                else:
                    t = 0.0
                    if (tmp1 <= 0.0):
                        s = 1.0
                    elif (b0 >= 0.0):
                        s = 0.0
                    else:
                        s = -b0 / a00
            else:  # region 1
                numer = a11 + b1 - a01 - b0
                if (numer <= 0.0):
                    s = 0.0
                    t = 1.0
                else:
                    denom = a00 - 2.0 * a01 + a11
                    if (numer >= denom):
                        s = 1.0
                        t = 0.0
                    else:
                        s = numer / denom
                        t = 1.0 - s
        closet = p0 + s * edge0 + t * edge1
        dis= (p  - closet).norm() 
        return dis,closet

    @ti.func
    def sdf_box(self, p,  min,  max):
        closest = ti.Vector([ti.math.clamp(p[0], min[0], max[0]), ti.math.clamp(p[1], min[1], max[1]), ti.math.clamp(p[2], min[2], max[2])])
        #https://iquilezles.org/articles/distfunctions/
        b = (max - min) * 0.5
        c = (max + min) * 0.5
        q = ti.Vector([abs(p[0] - c[0]), abs(p[1] - c[1]), abs(p[2] - c[2])]) - b
        dis = (ti.max(q,  ti.Vector([0.0,0.0,0.0]))).norm() + ti.math.min(ti.math.max(q[0], ti.math.max(q[1], q[2])), 0.0)
        return dis, closest
    
    @ti.func
    def intersect_triangle(self, origin, direction, index):
        hit_t     = DBL_MAX
        hit_pos   = ti.Vector([DBL_MAX,DBL_MAX,DBL_MAX])
        hit_bary  = ti.Vector([DBL_MAX,DBL_MAX]) #sometimes we wish bary to calculate texture_uv or tangent or sampled normal 

        # https:#www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
        p0 = self.tri_p1(index)
        p1 = self.tri_p2(index)
        p2 = self.tri_p3(index)
        #print(index, p0,p1,p2)

        ab = p1 - p0
        ac = p2 - p0
        n = ab.cross(ac)

        d = -direction.dot(n)
        ood = 1.0 / d
        ap = origin - p0

        t = ap.dot(n) * ood
        if (t >= 0.0):
            e = -direction.cross(ap)
            v = ac.dot(e) * ood
            if (v >= 0.0 and v <= 1.0):
                w = -ab.dot(e) * ood
                if (w >= 0.0 and v + w <= 1.0):
                    u = 1.0- v - w
                    hit_bary = ti.Vector([u, v])
                    hit_t = t
                    hit_pos = origin + t*direction
        return hit_t, hit_pos, hit_bary
    
    @ti.func
    def intersect_sphere(self, origin, direction, index):
        hit_t     = DBL_MAX
        hit_pos   = ti.Vector([DBL_MAX,DBL_MAX,DBL_MAX])
        hit_bary  = ti.Vector([DBL_MAX,DBL_MAX]) #sometimes we wish bary to calculate texture_uv or tangent or sampled normal 

        #   h1    h2          -->two hitpoint
        # o--*--p--*--->d     -->Ray
        #   \   |
        #    \  |
        #     \ |
        #      c              -->circle centre
        r      = self.radius(index)
        centre = self.center(index)
        oc     = centre - origin
        dis_oc_square = oc.dot(oc)
        dis_op        = direction.dot (oc)
        dis_cp        = ti.sqrt(dis_oc_square - dis_op * dis_op)
        if (dis_cp < r):
            # h1 is nearer than h2
            # because h1 = o + t*d
            # so  |ch| = radius = |c - d - t*d| = |oc - td|
            # so  radius*radius = (oc - td)*(oc -td) = oc*oc +t*t*d*d -2*t*(oc*d)
            #so d*d*t^2   -2*(oc*d)* t + (oc*oc- radius*radius) = 0

            #cal ax^2+bx+c = 0
            a = direction.dot(direction)
            b = -2.0 * dis_op
            c = dis_oc_square - r*r

            hit_t = (-b - ti.sqrt(b * b - 4.0 * a * c)) / 2.0 / a
            hit_pos = origin + hit_t * direction
        return hit_t, hit_pos, hit_bary
    
    @ti.func
    def intersect_quad(self, origin, direction, index):
        hit_t     = DBL_MAX
        hit_pos   = ti.Vector([DBL_MAX,DBL_MAX,DBL_MAX])
        hit_bary  = ti.Vector([DBL_MAX,DBL_MAX]) #sometimes we wish bary to calculate texture_uv or tangent or sampled normal 

        c  = self.center(index)
        v1 = self.quad_v1(index)
        v2 = self.quad_v2(index)
        v1 = v1 / v1.dot(v1)
        v2 = v2 / v2.dot(v2)
        n = v1.cross(v2)
        n.normalized()

        dt = direction.dot(n)
        PO = (c - origin)
        t = n.dot(PO) / dt
        if (t > 0.0):
            p = origin + direction * t
            vi = p - c
            a1 = v1.dot(vi)
            a2 = v2.dot(vi)
            if (a1 > -1.0 and a1 < 1.0 and a2 > -1.0 and a2 < 1.0):
                hit_t=   t
                hit_pos = p
        return hit_t, hit_pos, hit_bary
    
    @ti.func
    def intersect_box(self, origin, direction, index):
        hit_t     = DBL_MAX
        hit_pos   = ti.Vector([DBL_MAX,DBL_MAX,DBL_MAX])
        hit_bary  = ti.Vector([DBL_MAX,DBL_MAX]) #sometimes we wish bary to calculate texture_uv or tangent or sampled normal
        ret,hit_t,tmax = self.slabs(origin, direction, self.box_min(index), self.box_max(index))
        hit_pos = origin + hit_t * direction
        return hit_t, hit_pos, hit_bary 
    
    @ti.func
    def inside_triangle(self, index, p):
        #https:#www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
        p0 = self.tri_p1(index)
        p1 = self.tri_p2(index)
        p2 = self.tri_p3(index)
        edge0 = p1 - p0
        edge1 = p2 - p0
        n = edge0.cross(edge1)
        pa = p - p0
        #for hard edge, we need a tolerance
        ret = 0
        if (pa.dot(n) > -0.01):
            ret = 0
        else:
            ret = 1
        return ret
    
    @ti.func
    def inside_box(self, index, p):
        minv,maxv = self.aabb(index)
        ret = 0
        if (p[0] > minv[0] and p[1] > minv[1] and p[2] > minv[2] and p[0] < maxv[0] and p[1] < maxv[1] and p[2] < maxv[2]):
            ret = 1
        return ret

    @ti.func
    def coliison(self):
        #to be done
        a = 0