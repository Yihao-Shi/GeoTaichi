import taichi as ti

struct_candidate = ti.types.struct(
    t=gs.ti_int,  # 0: vf, 1: ee
    i0=gs.ti_int,
    i1=gs.ti_int,
    i2=gs.ti_int,
    i3=gs.ti_int,
)
self.candidate_data = struct_candidate.field(
    shape=(self.len_cd_candidate), needs_grad=False, layout=ti.Layout.SOA
)

struct_narrow_candidate = ti.types.struct(
    v0=gs.ti_int,
    v1=gs.ti_int,
    v2=gs.ti_int,
    v3=gs.ti_int,
    ctype=gs.ti_int,  # 0: vertex face; 1: edge edge
    x0=gs.ti_vec3,
    x1=gs.ti_vec3,
    x2=gs.ti_vec3,
    x3=gs.ti_vec3,
    dx0=gs.ti_vec3,
    dx1=gs.ti_vec3,
    dx2=gs.ti_vec3,
    dx3=gs.ti_vec3,
    # accd
    toi=gs.ti_float,
    is_col=gs.ti_int,  # 0: no collision; 1: collision; -1: unkown
    max_disp_mag=gs.ti_float,
    distance=gs.ti_float,
    # dvd
    penetration=gs.ti_float,
    n=gs.ti_vec3,
    pos=gs.ti_vec3,
    ga=gs.ti_int,
    gb=gs.ti_int,
)

self.narrow_candidate = struct_narrow_candidate.field(
    shape=(self.len_cd_candidate), needs_grad=False, layout=ti.Layout.SOA
)

struct_contact_aggregate = ti.types.struct(
    ctype=gs.ti_int,
    penetration=gs.ti_float,
    n=gs.ti_vec3,
    pos=gs.ti_vec3,
)
self.contact_aggregate = struct_contact_aggregate.field(
    shape=(self.solver.n_geoms_max, self.solver.n_geoms_max), needs_grad=False, layout=ti.Layout.SOA
)

struct_impact = ti.types.struct(
    pos=gs.ti_vec3,
    n=gs.ti_vec3,
    penetration=gs.ti_float,
    friction=gs.ti_float,
)
self.impact_data = struct_impact.field(shape=(self.len_cd_impact), needs_grad=True, layout=ti.Layout.SOA)

struct_impact_info = ti.types.struct(
    link_a=gs.ti_int,
    link_b=gs.ti_int,
    is_contact=gs.ti_int,
    i0=gs.ti_int,
    i1=gs.ti_int,
    i2=gs.ti_int,
    i3=gs.ti_int,
)
self.impact_info = struct_impact_info.field(shape=(self.len_cd_impact), needs_grad=False, layout=ti.Layout.SOA)