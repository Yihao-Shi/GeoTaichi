import taichi as ti
import math

from src.utils.constants import PI, MThreshold

k_B               = 1.38064852e-23
h                 = 6.62607015e-34
c                 = 299792458.0

########################### color function ###############################
#http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
@ti.func
def calc_matr_rgb_to_xyz(xy_r, xy_g, xy_b,XYZ_W):
    x_rgb = ti.Vector([xy_r.x, xy_g.x, xy_b.x])
    y_rgb = ti.Vector([xy_r.y, xy_g.y, xy_b.y])
    
    X_rgb = x_rgb / y_rgb
    Y_rgb = ti.Vector([1.0, 1.0, 1.0])
    Z_rgb = (ti.Vector([1.0, 1.0, 1.0]) - x_rgb - y_rgb ) / y_rgb
    
    S_rgb = ti.Matrix.rows([X_rgb,Y_rgb,Z_rgb]).inverse() @ XYZ_W[0]   
    return ti.Matrix.rows([S_rgb * X_rgb,S_rgb * Y_rgb,S_rgb * Z_rgb])


#https://en.wikipedia.org/wiki/Planck%27s_law#The_law
def Planck(Lambda, temperature):
    lambda_m = Lambda * 1.0e-9
    c_1L = 2.0 * h*c*c
    c_2  = h*c/k_B  
    numer = c_1L
    denom = pow(lambda_m,5.0) * (math.exp( c_2 / (lambda_m*temperature) ) - 1.0)
    value = numer / denom
    return value * 1.0e-9

@ti.func
def srgb_to_lrgb(srgb):
    ret = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        if srgb[i] < 0.04045:
            ret[i] = srgb[i] / 12.92
        else:
            ret[i] = pow((srgb[i] + 0.055) / 1.055, 2.4)
    return ret
   
@ti.func
def lrgb_to_srgb(lrgb):
    ret = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        if lrgb[i] < 0.0031308:
            ret[i] = lrgb[i] * 12.92
        else:
            ret[i] = 1.055 * pow(lrgb[i], 1.0 / 2.4) - 0.055
    return ti.math.clamp(ret, 0.0, 1.0)

@ti.func
def xyz_to_Yxy(xyz):
    ret = ti.Vector([0.0, 0.0, 0.0])
    coff = xyz[0] + xyz[1]+ xyz[2]
    if coff > 0.0:
        coff = 1.0 / coff
        ret  = ti.Vector([xyz[1], coff * xyz[0], coff * xyz[1]])
    return ret

@ti.func
def Yxy_to_xyz(yxy):
    ret = ti.Vector([0.0, 0.0, 0.0])
    if yxy[2] > 0.0:
        k = yxy[0] / yxy[2]
        ret = ti.Vector([k*yxy[1], yxy[0], k *(1.0 -yxy[1]-yxy[2])])
    return ret

#https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
@ti.func
def tone_ACES(x):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return ti.math.clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0, 1.0)


############algorithm##############
@ti.func
def mapToDisk(u1,u2):
    phi = 0.0
    r   = 0.0
    a   = 2.0 *u1 - 1.0
    b   = 2.0 *u2 - 1.0
    if (a > -b) :
        if (a > b):
            r = a
            phi = (PI / 4.0) * (b / a)
        else :
            r = b
            phi = (PI / 4.0) * (2.0 - a / b)
    else:
        if (a < b):
            r = -a
            phi = (PI / 4.0) * (4.0 + b / a)
        else:
            r = -b
            if b == 0.0:
                phi = 0.0
            else:
                phi = (PI / 4.0) * (6.0 - a / b)
    return r, phi

@ti.func
def CosineHemisphere_pdf(cosTheta):
    return max(0.01, cosTheta/PI)

@ti.func
def CosineSampleHemisphere( u1,  u2):
    r = ti.sqrt(u1)
    phi = 2.0*PI * u2
    p =   ti.Vector([0.0,0.0,0.0])
    p.x = r * ti.cos(phi)
    p.y = r * ti.sin(phi)
    p.z = ti.sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y))
    return p.normalized()

@ti.func
def CosineSampleHemisphere_pdf( u1,  u2):
    r = ti.sqrt(u1)
    phi = 2.0*PI * u2
    p =   ti.Vector([0.0,0.0,0.0])
    p.x = r * ti.cos(phi)
    p.y = r * ti.sin(phi)
    p.z = ti.sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y))
    p   = p.normalized()
    return p, CosineHemisphere_pdf(p.z)

@ti.func
def inverse_transform(dir, N):
    Normal   = N.normalized()
    Binormal = ti.Vector([0.0, 0.0, 0.0])
    if (abs(Normal.x) > abs(Normal.z)):
        Binormal.x = -Normal.y
        Binormal.y = Normal.x
        Binormal.z = 0.0
    else:
        Binormal.x = 0.0
        Binormal.y = -Normal.z
        Binormal.z = Normal.y
    Binormal = Binormal.normalized()
    Tangent  = Binormal.cross(Normal).normalized()
    return dir.x*Tangent + dir.y*Binormal + dir.z*Normal

@ti.func
def sqr(x):
     return x*x

@ti.func
def SchlickFresnel(u):
    m = ti.math.clamp(1.0-u, 0.0, 1.0)
    m2 = m*m
    return m2*m2*m

@ti.func
def GTR1(NDotH,  a):
    ret =1.0/ PI
    if (a < 1.0):
        a2 = a*a
        t = 1.0 + (a2-1.0)*NDotH*NDotH
        ret = (a2-1.0) / (PI*ti.log(a2)*t)
    return ret 

@ti.func
def GTR2(NDotH,  a):
    a2 = a*a
    t = 1.0 + (a2-1.0)*NDotH*NDotH
    return a2 / (PI * t*t)

@ti.func
def smithG_GGX(NDotv,  alphaG):
    a = alphaG*alphaG
    b = NDotv*NDotv
    return 1.0/(NDotv + ti.sqrt(a + b - a*b))

@ti.func
def refract(InRay, N,  eta):
    suc  = -1.0 
    N_DOT_I = N.dot(InRay)
    k = 1.0 - eta * eta * (1.0 - N_DOT_I * N_DOT_I)
    R = ti.Vector([0.0,0.0,0.0])
    if k > 0.0:
        R = eta * InRay - (eta * N_DOT_I + ti.sqrt(k)) * N
        suc = 1.0
    return R,suc

@ti.func
def schlick(cosine,  index_of_refraction):
    r0 = (1.0 - index_of_refraction) / (1.0 + index_of_refraction)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0)

@ti.func
def powerHeuristic( a,  b):
    t = a* a
    return t / (b*b + t)

@ti.func
def offset_ray(p,  n):
    int_scale   = 256.0
    float_scale = 1.0 / 2048.0
    origin      = 1.0 / 256.0
    ret         = ti.Vector([0.0, 0.0, 0.0])
    for k in ti.static(range(3)):
        i_of = int(int_scale * n[k])
        i_p  = ti.bit_cast(p[k], ti.i32)
        if p[k] < 0.0:
            i_p = i_p - i_of
        else:
            i_p = i_p + i_of
        f_p = ti.bit_cast(i_p, ti.f32)
        if abs(p[k]) < origin:
            ret[k] = p[k] + float_scale * n[k]
        else:
            ret[k] = f_p
    return ret

# sometimes 3d model software will do normal smoothing,
# that will change the true geometry normal,so we use geometry normal as a ref
@ti.func
def faceforward(n,  i,  nref):
    return ti.math.sign(i.dot(nref)) * n

@ti.func
def srgb_to_lrgb(srgb):
    ret = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        if srgb[i] < 0.04045:
            ret[i] = srgb[i] / 12.92
        else:
            ret[i] = pow((srgb[i] + 0.055) / 1.055, 2.4)
    return ret

# https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
@ti.func
def get_glass_ior(Lambda):
    Lambda      = Lambda/1000.0
    Lambda2     = Lambda*Lambda
    return      ti.sqrt(1.0 + 1.03961212 * Lambda2/ (Lambda2 -0.00600069867 )+ 0.231792344 * Lambda2/ (Lambda2 -0.0200179144 ) + 1.01046945 * Lambda2/ (Lambda2 -103.560653))

@ti.func
def max_component( v):
    return max(v.z, max(v[0], v.y) )

@ti.func
def min_component( v):
    return min(v.z, min(v[0], v.y) )

@ti.func
def slabs(origin, direction, minv, maxv):
    # most effcient algrithm for ray intersect aabb 
    # en vesrion: https://www.researchgate.net/publication/220494140_An_Efficient_and_Robust_Ray-Box_Intersection_Algorithm
    ret  = 1
    tmin = 0.0
    tmax = MThreshold
    
    for i in ti.static(range(3)):
        if abs(direction[i]) < 0.000001:
            if ( (origin[i] < minv[i]) | (origin[i] > maxv[i])):
                ret = 0
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
    return ret

#https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
@ti.kernel
def tone_map(exposure:ti.f32, input:ti.template(), output:ti.template()):
    for i,j in output:   
        output[i,j] =  lrgb_to_srgb(tone_ACES(input[i,j]*exposure))
