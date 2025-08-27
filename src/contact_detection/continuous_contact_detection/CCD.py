import taichi as ti


@ti.func
def det3x2(self, b, c, d):
    return (
        b[0] * c[1] * d[2]
        + c[0] * d[1] * b[2]
        + d[0] * b[1] * c[2]
        - d[0] * c[1] * b[2]
        - c[0] * b[1] * d[2]
        - b[0] * d[1] * c[2]
    )

@ti.func
def point_inside_tet(self, v0, v1, v2, v3, p):
    a = v0 - p
    b = v1 - p
    c = v2 - p
    d = v3 - p
    detA = self.det3x2(b, c, d)
    detB = self.det3x2(a, c, d)
    detC = self.det3x2(a, b, d)
    detD = self.det3x2(a, b, c)
    ret0 = detA > 0.0 and detB < 0.0 and detC > 0.0 and detD < 0.0
    ret1 = detA < 0.0 and detB > 0.0 and detC < 0.0 and detD > 0.0

    is_col = ret0 or ret1
    return is_col

@ti.func
def segment_intersect_triangle(self, e0, e1, A, B, C):
    Norm = (e1 - e0).norm()
    Dir = (e1 - e0) / Norm
    Origin = e0

    E1 = B - A
    E2 = C - A
    N = E1.cross(E2)

    det = -Dir.dot(N)
    invdet = 1.0 / det
    AO = Origin - A
    DAO = AO.cross(Dir)
    u = E2.dot(DAO) * invdet
    v = -E1.dot(DAO) * invdet
    t = AO.dot(N) * invdet

    t = t / Norm

    is_col = det >= 1e-6 and t > 0.0001 and t < 0.9999 and u >= 0.0 and v >= 0.0 and (u + v) <= 1.0
    return is_col, t, u, v

@ti.func
def point_plane_distance_normal(self, p, origin, normal):
    point_to_plane = (p - origin).dot(normal)
    d = point_to_plane / normal.norm()
    return d**2

@ti.func
def point_plane_distance(self, p, t0, t1, t2):
    return self.point_plane_distance_normal(p, t0, (t1 - t0).cross(t2 - t0))

@ti.func
def cd_check_penetration(self, i, ctype):

    v0 = self.narrow_candidate[i].v0
    v1 = self.narrow_candidate[i].v1
    v2 = self.narrow_candidate[i].v2
    v3 = self.narrow_candidate[i].v3
    # min_distance = 0.001

    if ctype == 0:  # vertex face
        p_t1 = self.solver.verts_state[v0].pos
        t0_t1 = self.solver.verts_state[v1].pos
        t1_t1 = self.solver.verts_state[v2].pos
        t2_t1 = self.solver.verts_state[v3].pos

        t_geom1_idx = self.solver.verts_info[v1].geom_idx
        t_center_b = self.solver.verts_state[v1].center_pos

        # print(t0_t1, t1_t1, t2_t1, t_center_b, p_t1)
        is_col = self.point_inside_tet(t0_t1, t1_t1, t2_t1, t_center_b, p_t1)

        if is_col == 1:
            penetration = ti.sqrt(self.point_plane_distance(p_t1, t0_t1, t1_t1, t2_t1))

            if penetration > self.narrow_candidate[i].penetration:

                self.narrow_candidate[i].ctype = ctype
                self.narrow_candidate[i].is_col = is_col
                self.narrow_candidate[i].penetration = penetration

                n = (t1_t1 - t0_t1).cross(t2_t1 - t0_t1)
                sign = (n.dot(t0_t1 - t_center_b) > 0) * 2 - 1
                self.narrow_candidate[i].n = n * sign / n.norm()
                self.narrow_candidate[i].pos = p_t1

                self.narrow_candidate[i].ga = self.solver.verts_info[v0].geom_idx
                self.narrow_candidate[i].gb = self.solver.verts_info[v1].geom_idx

    elif ctype == 1:  # edge edge
        ea0_t1 = self.solver.verts_state[v0].pos
        ea1_t1 = self.solver.verts_state[v1].pos
        eb0_t1 = self.solver.verts_state[v2].pos
        eb1_t1 = self.solver.verts_state[v3].pos

        t_geom2_idx = self.solver.verts_info[v2].geom_idx
        t_center_b = self.solver.verts_state[v2].center_pos

        is_col, t, u, v = self.segment_intersect_triangle(ea0_t1, ea1_t1, eb0_t1, eb1_t1, t_center_b)

        # if ea0_t1[2] == ea1_t1[2] and eb0_t1[2] == eb1_t1[2]:
        #     if ti.abs(ea1_t1[2] - eb0_t1[2]) == 1.0 and ti.min(ea1_t1[2], eb0_t1[2]) == 0.0:
        #         print("!!!!!", i)
        #         print(ea0_t1, ea1_t1, eb0_t1, eb1_t1)
        #         print(t_center_b, t, u, v)
        # if self.narrow_candidate[i].is_col:
        #     print("-----", i)
        #     print(ea0_t1, ea1_t1, eb0_t1, eb1_t1)
        #     print(t_center_b, t)
        #     print(self.point_plane_distance(ea1_t1, eb0_t1, eb1_t1, t_center_b))

        if is_col == 1:

            n = (ea1_t1 - ea0_t1).cross(eb1_t1 - eb0_t1)
            # sign = (n.dot(self.narrow_candidate[i].pos - t_center_b) > 0) * 2 - 1
            sign = (n.dot(eb0_t1 - t_center_b) > 0) * 2 - 1
            n = n * sign / n.norm()
            # TODO: sign here

            pos = ea0_t1 * (1 - t) + ea1_t1 * t
            penetration = -n.dot(pos - eb0_t1)
            # print("edge col!", penetration)

            if penetration > self.narrow_candidate[i].penetration:

                self.narrow_candidate[i].ctype = ctype
                self.narrow_candidate[i].is_col = is_col
                self.narrow_candidate[i].penetration = penetration

                self.narrow_candidate[i].pos = pos
                self.narrow_candidate[i].n = n

                self.narrow_candidate[i].ga = self.solver.verts_info[v0].geom_idx
                self.narrow_candidate[i].gb = self.solver.verts_info[v2].geom_idx

            # if self.narrow_candidate[i].penetration > 0:
            #     print("===edge", self.narrow_candidate[i].ga, self.narrow_candidate[i].gb,
            #         sign, self.narrow_candidate[i].n, n, self.narrow_candidate[i].pos - t_center_b)
            #     print("points", eb1_t1, eb0_t1, t_center_b)

            # print(t, self.narrow_candidate[i].n,
            #     self.narrow_candidate[i].ga, self.narrow_candidate[i].gb,
            #     self.narrow_candidate[i].penetration)

@ti.func
def cd_add_face_and_dcd(self, fa, fb):
    idx = self.cd_counter[0]
    self.cd_counter[0] += 1
    # idx = ti.atomic_add(self.cd_counter[0], 1)
    self.narrow_candidate[idx].is_col = 0
    self.narrow_candidate[idx].penetration = -1

    for j in ti.static(range(3)):
        self.narrow_candidate[idx].v0 = self.solver.faces_info[fa].verts_idx[j % 3]
        self.narrow_candidate[idx].v1 = self.solver.faces_info[fb].verts_idx[0]
        self.narrow_candidate[idx].v2 = self.solver.faces_info[fb].verts_idx[1]
        self.narrow_candidate[idx].v3 = self.solver.faces_info[fb].verts_idx[2]
        self.cd_check_penetration(idx, ctype=0)

    for j in ti.static(range(3)):
        self.narrow_candidate[idx].v0 = self.solver.faces_info[fb].verts_idx[j % 3]
        self.narrow_candidate[idx].v1 = self.solver.faces_info[fa].verts_idx[0]
        self.narrow_candidate[idx].v2 = self.solver.faces_info[fa].verts_idx[1]
        self.narrow_candidate[idx].v3 = self.solver.faces_info[fa].verts_idx[2]

        self.cd_check_penetration(idx, ctype=0)

    # edge edge
    for j1 in ti.static(range(3)):
        for j2 in ti.static(range(3)):
            self.narrow_candidate[idx].v0 = self.solver.faces_info[fa].verts_idx[j1 % 3]
            self.narrow_candidate[idx].v1 = self.solver.faces_info[fa].verts_idx[(j1 + 1) % 3]
            self.narrow_candidate[idx].v2 = self.solver.faces_info[fb].verts_idx[j2 % 3]
            self.narrow_candidate[idx].v3 = self.solver.faces_info[fb].verts_idx[(j2 + 1) % 3]
            self.cd_check_penetration(idx, ctype=1)

@ti.func
def cd_add_face(self, fa, fb):
    # !!
    # print("fa, fb", fa, fb)
    for i in ti.static(range(3)):
        self.cd_add_vf(self.solver.faces[fa].verts_idx[i], fb)
        self.cd_add_vf(self.solver.faces[fb].verts_idx[i], fa)

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            self.cd_add_ee(fa, fb, i, j)

@ti.kernel
def cd_aggregate(self):

    for i in range(self.contact_aggregate.shape[0]):
        for j in range(self.contact_aggregate.shape[1]):
            self.contact_aggregate[i, j].ctype = -1

    for i in range(self.cd_counter[0]):
        if self.narrow_candidate[i].is_col == 1:
            ga = self.narrow_candidate[i].ga
            gb = self.narrow_candidate[i].gb
            ctype = self.narrow_candidate[i].ctype

            n = self.narrow_candidate[i].n
            pos = self.narrow_candidate[i].pos
            penetration = self.narrow_candidate[i].penetration

            if ga == gb:
                print("warning ga == gb", ga, gb, ctype)
            if ga >= gb:
                ga, gb = gb, ga
                n = n * -1

            if self.contact_aggregate[ga, gb].ctype == -1:
                self.contact_aggregate[ga, gb].ctype = ctype

                self.contact_aggregate[ga, gb].n = n
                self.contact_aggregate[ga, gb].pos = pos
                self.contact_aggregate[ga, gb].penetration = penetration

            elif self.contact_aggregate[ga, gb].ctype == 0:  # vertex face
                if (
                    self.narrow_candidate[i].ctype == 0
                    and self.contact_aggregate[ga, gb].penetration > self.narrow_candidate[i].penetration
                ):
                    self.contact_aggregate[ga, gb].n = n
                    self.contact_aggregate[ga, gb].pos = pos
                    self.contact_aggregate[ga, gb].penetration = penetration

            else:  # edge edge
                if (
                    self.narrow_candidate[i].ctype == 0
                    or self.contact_aggregate[ga, gb].penetration > self.narrow_candidate[i].penetration
                ):
                    self.contact_aggregate[ga, gb].ctype = ctype
                    self.contact_aggregate[ga, gb].n = n
                    self.contact_aggregate[ga, gb].pos = pos
                    self.contact_aggregate[ga, gb].penetration = penetration

def collision_detection(self):
    from genesis.utils.tools import create_timer

    timer = create_timer(name="solve_quadratic", level=4, ti_sync=True, skip_first_call=True)
    self.cd_init()
    timer.stamp("cd_init")
    self.refit_func()
    timer.stamp("refit_func")
    self.cd_tree_phase()  # 18 ms
    timer.stamp("cd_tree_phase")
    self.cd_aggregate()
    timer.stamp("cd_aggregate")
    # print("overlapped_face, overlapped_node", self.cd_counter[0], self.cd_counter[1])

    # exit()
    # self.cd_candidate_phase() # not too much
    # print("self.cd_counter 2", self.cd_counter)
    # self.cd_impact_phase() # 1 ms
    # print("self.cd_counter 3", self.cd_counter)

@ti.kernel
def compute_constraint_system(self, n_con: int, n_dof: int):
    # TODO TEST
    for jd1 in range(n_dof):
        self.solver.dof_state.qf_smooth[jd1] = 0.1

    for ic1 in range(n_con):
        for ic2 in range(n_con):
            self.constraint_A[ic1, ic2] = 0.0

    for ic1 in range(n_con):
        self.constraint_b[ic1] = 0.0

    for ic1 in range(n_con):
        for ic2 in range(n_con):
            for jd1 in range(n_dof):
                for jd2 in range(n_dof):
                    self.constraint_A[ic1, ic2] += (
                        self.con_jac[ic1, jd1] * self.solver.mass_mat_inv[jd1, jd2] * self.con_jac[ic2, jd2]
                    )

    for ic1 in range(n_con):
        for jd1 in range(n_dof):
            for jd2 in range(n_dof):
                self.constraint_b[ic1] += (
                    self.con_jac[ic1, jd1]
                    * self.solver.mass_mat_inv[jd1, jd2]
                    * self.solver.dof_state.qf_smooth[jd2]
                )

    for ic1 in range(n_con):
        self.constraint_b[ic1] -= self.con_aref[ic1]

    for ic1 in range(n_con):
        self.x[ic1] = 0.0

@ti.kernel
def solve_system(self, Gammak: float, n_con: int, n_dof: int):
    for it in ti.static(range(self.n_iters)):
        # x_kplus1 = xk - Gammak*grf(A,xk,b)
        ## A.dot(x) - b
        for ni in range(n_con):
            self.Ax_b[ni] = -self.constraint_b[ni]
        for ni in range(n_con):
            for pi in range(n_con):
                self.Ax_b[ni] += self.constraint_A[ni, pi] * self.x[pi]
        ## xgrad = A.T.dot(Ax_b)
        for pi in range(n_con):
            self.x1[pi] = self.x[pi]
            self.xgrad[pi] = 0
        for ni in range(n_con):
            for pi in range(n_con):
                self.xgrad[pi] += self.constraint_A[ni, pi] * self.Ax_b[ni]
        ## x1 = xk - Gammak * xgrad
        for pi in range(n_con):
            self.x1[pi] -= Gammak * self.xgrad[pi]

        # projection
        for pi in range(n_con):
            self.x[pi] = ti.math.sign(self.x1[pi]) * ti.max(0.0, ti.abs(self.x1[pi]) - self.lamda)

@ti.kernel
def project_constraint_force(self, n_con: int, n_dof: int):
    for jd1 in range(n_dof):
        self.solver.dof_state.qf_constraint[jd1] = 0.0

    for ic1 in range(n_con):
        for jd1 in range(n_dof):
            self.solver.dof_state.qf_constraint[jd1] += self.con_jac[ic1, jd1] * self.x[ic1]

@ti.kernel
def cd_init(self):
    for i in range(self.cd_counter.shape[0]):
        self.cd_counter[i] = 0

@ti.kernel
def cd_tree_phase(self):
    for pi in range(self.collision_pairs.shape[0]):
        ga = self.collision_pairs[pi][0]
        gb = self.collision_pairs[pi][1]

        na = 2 * self.solver.geoms_info[ga].face_start
        nb = 2 * self.solver.geoms_info[gb].face_start

        head = tail = 0

        self.cd_node_que[ga, gb, tail, 0] = na
        self.cd_node_que[ga, gb, tail, 1] = nb
        tail += 1

        # print("na, nb", na, nb)

        box = self.node_data[na].box
        # print("box", na, box, self.box_data[box].umax, self.box_data[box].umin)

        box = self.node_data[nb].box
        # print("box", nb, box, self.box_data[box].umax, self.box_data[box].umin)

        while head < tail:
            self.cd_counter[1] += 1
            _h = head % self.len_cd_que
            na = self.cd_node_que[ga, gb, _h, 0]
            nb = self.cd_node_que[ga, gb, _h, 1]
            # print("na, nb", na, nb)

            head += 1
            if not self._box.overlap(self.node_data[na].box, self.node_data[nb].box):
                continue

            if self.node_data[na].face != -1 and self.node_data[nb].face != -1:
                # self.cd_add_face(self.node_data[na].face, self.node_data[nb].face)
                self.cd_add_face_and_dcd(self.node_data[na].face, self.node_data[nb].face)

            elif self.node_data[na].face != -1:
                self.cd_node_que[ga, gb, tail % self.len_cd_que, 0] = na
                self.cd_node_que[ga, gb, tail % self.len_cd_que, 1] = self.node_data[nb].left
                tail += 1
                self.cd_node_que[ga, gb, tail % self.len_cd_que, 0] = na
                self.cd_node_que[ga, gb, tail % self.len_cd_que, 1] = self.node_data[nb].right
                tail += 1
            else:
                self.cd_node_que[ga, gb, tail % self.len_cd_que, 0] = self.node_data[na].left
                self.cd_node_que[ga, gb, tail % self.len_cd_que, 1] = nb
                tail += 1
                self.cd_node_que[ga, gb, tail % self.len_cd_que, 0] = self.node_data[na].right
                self.cd_node_que[ga, gb, tail % self.len_cd_que, 1] = nb
                tail += 1


