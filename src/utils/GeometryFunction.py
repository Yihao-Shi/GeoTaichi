import taichi as ti

from src.utils.constants import PI
from src.utils.Quaternion import RodriguesRotationMatrix
from src.utils.ScalarFunction import clamp, sgn
from src.utils.TypeDefination import vec3f
from src.utils.VectorFunction import dot2


# refers to https://stackoverflow.com/questions/540014/compute-the-area-of-intersection-between-a-circle-and-a-triangle
@ti.func
def SphereTriangleIntersectionArea(position, radius, vertice1, vertice2, vertice3, norm):
	area = 0.
	processSegment(area, position, radius, norm, vertice3, vertice2)
	processSegment(area, position, radius, norm, vertice2, vertice1)
	processSegment(area, position, radius, norm, vertice1, vertice3)
	return area 


@ti.func
def processSegment(area: ti.template(), position, radius, norm, initialVertex, finalVertex):
	segmentDisplacement = finalVertex - initialVertex
	centerToInitialDisplacement = initialVertex - position

	segmentLength = segmentDisplacement.norm()
	leftX = centerToInitialDisplacement.dot(segmentDisplacement) / segmentLength
	rightX = leftX + segmentLength
	outer_normal = segmentDisplacement.cross(centerToInitialDisplacement)
	y = sgn(outer_normal.dot(norm)) * outer_normal.norm() / segmentLength
	processSegmentStandardGeometry(area, radius, leftX, rightX, y)


@ti.func
def processSegmentStandardGeometry(area: ti.template(), radius, leftX, rightX, y):
	if y * y > radius * radius:
		processNonIntersectingRegion(area, radius, leftX, rightX, y)
	else:
		intersectionX = ti.sqrt(radius * radius - y * y)
		if leftX < -intersectionX:
			leftRegionRightEndpoint = ti.min(-intersectionX, rightX)
			processNonIntersectingRegion(area, radius, leftX, leftRegionRightEndpoint, y)
		if intersectionX < rightX:
			rightRegionLeftEndpoint = ti.max(intersectionX, leftX)
			processNonIntersectingRegion(area, radius, rightRegionLeftEndpoint, rightX, y)
		middleRegionLeftEndpoint = ti.max(-intersectionX, leftX)
		middleRegionRightEndpoint = ti.min(intersectionX, rightX)
		middleRegionLength = ti.max(middleRegionRightEndpoint - middleRegionLeftEndpoint, 0)
		processIntersectingRegion(area, middleRegionLength, y)


@ti.func
def processNonIntersectingRegion(area: ti.template(), radius, leftX, rightX, y):
	initialTheta = ti.atan2(y, leftX)
	finalTheta = ti.atan2(y, rightX)
	deltaTheta = finalTheta - initialTheta
	if deltaTheta < -PI:
		deltaTheta += 2 * PI
	elif deltaTheta > PI:
		deltaTheta -= 2 * PI
	area += 0.5 * radius * radius * deltaTheta


@ti.func
def processIntersectingRegion(area: ti.template(), length, y):
	area -= 0.5 * length * y


@ti.func
def IsLineIntersectTriangle(linePoint1, linePoint2, _planePoint1, _planePoint2, _planePoint3):
	is_intersection = 0
	
	u =_planePoint3 - _planePoint1    
	v =_planePoint2 - _planePoint1
	n = u * v
	dir = linePoint2 - linePoint1
	w0 = linePoint1 - _planePoint1
	a = -n.dot(w0)
	b = n.dot(dir)

	if ti.abs(b) > 0.:
		r = a / b
		if r > 0.0:
			I = linePoint1 + dir * r         

			uu = u.dot(u)
			uv = u.dot(v)
			vv = v.dot(v)
			w = I - _planePoint1
			wu = w.dot(u)
			wv = w.dot(v)
			D = uv * uv - uu * vv

			s = (uv * wv - vv * wu) / D
			if s < 0.0 or s > 1.0:     
				is_intersection = 1
			t = (uv * wu - uu * wv) / D
			if t >= 0.0 and (s + t) <= 1.0:  
				is_intersection = 1
	return is_intersection


@ti.func
def IntersectRaySphere(_P, _D, _SphereCenter, _dRadius, _pLength, _pIntersection):
	is_intersection = 0
	m = _P - _SphereCenter
	b = m.dot(_D)
	dC = m.dot(m) - _dRadius * _dRadius
	if dC <= 0 or b <= 0: 
		dDiscr = b * b - dC
		if dDiscr >= 0: 
			_pLength = - b
			if _pLength < 0: _pLength = 0
			_pIntersection = _P + _D * _pLength
			is_intersection = 1
	return is_intersection


# refers to Arvo, A Simple Method for Box-Sphere Intersection Testing, Graphics Gems, pp. 247-250, 1993
@ti.func
def IsCircleIntersectRectangle(circle_center, radius, recenter, hlength):
	v = ti.abs(circle_center - recenter)
	u = ti.max(v - hlength, 0.)
	return u.dot(u) < radius * radius


@ti.func
def PointProjectionToRectangle(circle_center, recenter, norm, length, width, height):
    p = circle_center - recenter
    rotation_matrix = RodriguesRotationMatrix(vec3f(0, 0, 1), norm)
    localp = rotation_matrix.transpose() @ p

    dist1 = localp.dot(vec3f(1, 0, 0))
    dist2 = localp.dot(vec3f(0, 1, 0))
    dist3 = localp.dot(vec3f(0, 0, 1))
    dist1 = clamp(-length, length, dist1)
    dist2 = clamp(-width, width, dist2)
    dist3 = clamp(-height, height, dist3)
    q = dist1 * vec3f(1, 0, 0) + dist2 * vec3f(0, 1, 0) + dist3 * vec3f(0, 0, 1)
    return rotation_matrix @ q + recenter


@ti.func
def SpheresIntersectionVolume(center1, center2, rad1, rad2):
	dR = ti.max(rad1, rad2)
	dr = ti.min(rad1, rad2)
	dDistance = (center1 - center2).norm()
    
	volume = 0.
	if dDistance + dr < dR: 
		volume = 4./3. * PI * dr * dr * dr
	else:
		db = (dr * dr - dR * dR + dDistance * dDistance) / (2. * dDistance)
		if db >= 0.:
			dh1 = dr - db
			dh2 = dR - dDistance + db
			volume = PI / 3. * dh1 * dh1 * (3 * dr - dh1) + PI / 3. * dh2 * dh2 * (3. * dR - dh2)
		else:
			dh1 = dr - ti.abs(db)
			dh2 = dR - dDistance - ti.abs(db)
			volume = 4. / 3. * PI * dr * dr * dr - PI / 3. * dh1 * dh1 * (3 * dr - dh1) + PI / 3. * dh2 * dh2 * (3 * dR - dh2)
	return volume	


@ti.func
def DistanceFromPointToSegment(point, vertice1, vertice2):
	return ((point - vertice1).cross(point - vertice2)).norm() / (vertice2 - vertice1).norm()


@ti.func
def LineLineIntersection(p1L1, p2L1, p1L2, p2L2):
	dirVecL1 = p2L1 - p1L1
	dirVecL2 = p2L2 - p1L2
	denomX = dirVecL1[1] * dirVecL2[0] - dirVecL2[1] * dirVecL1[0]
	denomY = dirVecL1[0] * dirVecL2[1] - dirVecL2[0] * dirVecL1[1]
	denomZ = dirVecL1[1] * dirVecL2[2] - dirVecL2[1] * dirVecL1[2]
	x = p1L1[0] * dirVecL1[1] * dirVecL2[0] - p1L2[0] * dirVecL2[1] * dirVecL1[0] - p1L1[1] * dirVecL1[0] * dirVecL2[0] + p1L2[1] * dirVecL2[0] * dirVecL1[0]
	y = p1L1[1] * dirVecL1[0] * dirVecL2[1] - p1L2[1] * dirVecL2[0] * dirVecL1[1] - p1L1[0] * dirVecL1[1] * dirVecL2[1] + p1L2[0] * dirVecL2[1] * dirVecL1[1]
	z = p1L1[2] * dirVecL1[1] * dirVecL2[2] - p1L2[2] * dirVecL2[1] * dirVecL1[2] - p1L1[1] * dirVecL1[2] * dirVecL2[2] + p1L2[1] * dirVecL2[2] * dirVecL1[2]
	
	if denomX == 0:
		if x != 0:
			x = 0/0
	else: 
		x /= denomX

	if denomY == 0:
		if y != 0:	
			y = 0/0
	else: 
		y /= denomY

	if denomZ == 0:
		if z != 0:
			z = 0/0
	else:
		z /= denomZ

	return vec3f(x, y, z)

# Refers to https://iquilezles.org/articles/distfunctions/
@ti.func
def DistanceFromPointToTriangle(point, vertice1, vertice2, vertice3):
    ba = vertice2 - vertice1
    cb = vertice3 - vertice2
    ac = vertice1 - vertice3
    pa = point - vertice1
    pb = point - vertice2
    pc = point - vertice3
    norm = ba.cross(ac)

    A = sgn((ba.cross(norm).dot(pa))) + sgn((cb.cross(norm).dot(pb))) + sgn((ac.cross(norm).dot(pc)))
    return ti.sqrt(ti.min(dot2(ba * clamp(0., 1., ba.dot(pa) / dot2(ba)) - pa), dot2(cb * clamp(0., 1., cb.dot(pb) / dot2(cb)) - pb), dot2(ac * clamp(0., 1., ac.dot(pc) / dot2(ac)) - pc))) \
	       if A < 2. else ti.sqrt(norm.dot(pa) * norm.dot(pa) / dot2(norm))


@ti.func
def DistanceFromPointToRectangle(point, vertice1, vertice2, vertice3, vertice4):
    ba = vertice2 - vertice1
    cb = vertice3 - vertice2
    dc = vertice4 - vertice3
    ad = vertice1 - vertice4
    pa = point - vertice1
    pb = point - vertice2
    pc = point - vertice3
    pd = point - vertice4
    norm = ba.cross(ad)

    A = sgn((ba.cross(norm).dot(pa))) + sgn((cb.cross(norm).dot(pb))) + sgn((dc.cross(norm).dot(pc))) + sgn((ad.cross(norm).dot(pd)))
    return ti.sqrt(ti.min(dot2(ba * clamp(0., 1., ba.dot(pa) / dot2(ba)) - pa), dot2(cb * clamp(0., 1., cb.dot(pb) / dot2(cb)) - pb), 
						  dot2(dc * clamp(0., 1., dc.dot(pc) / dot2(dc)) - pc), dot2(ad * clamp(0., 1., ad.dot(pd) / dot2(ad)) - pd))) \
	       if A < 3. else ti.sqrt(norm.dot(pa) * norm.dot(pa) / dot2(norm))


# Refers to Philip J. Schneider, and David H. Eberly. Geometric Tools for Computer Graphics. Morgan Kaufmann, 2002
@ti.func
def DistanceFromPointToTriangle2(point, vertice1, vertice2, vertice3):
	vEdge0 = vertice2 - vertice1
	vEdge1 = vertice3 - vertice1
	vD = vertice1 - point

	a = vEdge0.dot(vEdge0)
	b = vEdge0.dot(vEdge1)
	c = vEdge1.dot(vEdge1)
	d = vEdge0.dot(vD)
	e = vEdge1.dot(vD)
	f = dot2(vD)

	det = a * c - b * b
	s = b * e - c * d
	t = b * d - a * e

	if s + t < det:
		if s < 0.:
			if t < 0.:
				if d < 0.:
					s = clamp(0., 1., -d / a)
					t = 0.
				else:
					s = 0.
					t = clamp(0., 1., -e / c)
			else:
				s = 0.
				t = clamp(0., 1., -e / c)
		elif t < 0.:
			s = clamp(0., 1., -d / a)
			t = 0.
		else:
			invDet = 1. / det
			s *= invDet
			t *= invDet
	else:
		if s < 0.:
			tmp0 = b + d
			tmp1 = c + e
			if tmp1 > tmp0:
				numer = tmp1 - tmp0
				denom = a - 2. * b + c
				s = clamp(0., 1., numer / denom)
				t = 1 - s
			else:
				t = clamp(0., 1., -e / c)
				s = 0
		elif t < 0.:
			if a+d > b+e:
				numer = c + e - b - d
				denom = a - 2. * b + c
				s = clamp(0., 1., numer / denom)
				t = 1 - s
			else:
				s = clamp(0., 1., -e / c)
				t = 0
		else:
			numer = c + e - b - d
			denom = a - 2. * b + c
			s = clamp(0., 1., numer / denom)
			t = 1.0 - s
	return ti.sqrt(a * s * s + 2 * b * s * t + c * t * t + 2 * d * s + 2 * e * t + f)


@ti.func
def PointProjectionDistance(position, center, norm):
	return (position - center).dot(norm)


@ti.func
def ProjectionPointInPlane(position, norm, distance):
	return position - norm * distance


@ti.func
def IsPointInTriangle(projection_point, vertice1, vertice2, vertice3):
    edge21 = vertice2 - vertice1
    edge31 = vertice3 - vertice1
    W = projection_point - vertice1

    d00 = edge21.dot(edge21)
    d01 = edge21.dot(edge31)
    d11 = edge31.dot(edge31)
    d20 = W.dot(edge21)
    d21 = W.dot(edge31)
    invDenom = 1.0 / (d00 * d11 - d01 * d01)
    gamma = (d11 * d20 - d01 * d21) * invDenom
    betta = (d00 * d21 - d01 * d20) * invDenom
    alpha = 1. - gamma - betta
    return (gamma > 0 and gamma < 1) and (alpha > 0 and alpha < 1) and (betta > 0 and betta < 1)


@ti.func
def IsPointInTriangle2(projection_point, vertice1, vertice2, vertice3):
    u = (vertice1 - projection_point).cross(vertice2 - projection_point)
    v = (vertice2 - projection_point).cross(vertice3 - projection_point)
    w = (vertice3 - projection_point).cross(vertice1 - projection_point)
    return u.dot(v) >= 0. and u.dot(w) >= 0.


@ti.func
def IsSphereIntersectTriangle(bound_beg, bound_end, vertice1, vertice2, vertice3, norm, position, radius):
	status = 0
	point = vec3f(0/0, 0/0, 0/0)

	if position[0] <= bound_beg[0] - radius and position[1] <= bound_beg[1] - radius and position[2] <= bound_beg[2] - radius and \
	   position[0] >= bound_end[0] + radius and position[1] >= bound_end[1] + radius and position[2] >= bound_end[2] + radius:
		center = (vertice1 + vertice2 + vertice3) / 3.0
		ppd = PointProjectionDistance(position, center, norm)
		if ti.abs(ppd) < radius:
			A = ProjectionPointInPlane(position, norm, ppd)
			if IsPointInTriangle(A, vertice1, vertice2, vertice3):
				status = 1
				point = A
			else:
				edge21 = vertice2 - vertice1
				edge32 = vertice3 - vertice2
				edge13 = vertice1 - vertice3
				lc1 = clamp(0., 1., (A - vertice1).dot(edge21) / dot2(edge21))
				lc2 = clamp(0., 1., (A - vertice2).dot(edge32) / dot2(edge32))
				lc3 = clamp(0., 1., (A - vertice3).dot(edge13) / dot2(edge13))
				C1IsVertice = lc1 == 0 or lc1 == 1
				C2IsVertice = lc2 == 0 or lc2 == 1
				C3IsVertice = lc3 == 0 or lc3 == 1
				C1 = vertice1 + lc1 * edge21
				C2 = vertice2 + lc2 * edge32
				C3 = vertice3 + lc3 * edge13
				sqrLength1 = dot2(C1 - position)
				sqrLength2 = dot2(C2 - position)
				sqrLength3 = dot2(C3 - position)
				if ti.min({ sqrLength1, sqrLength2, sqrLength3 }) < radius * radius:
					if sqrLength1 <= sqrLength2 and sqrLength1 <= sqrLength3:
						status = 3 if C1IsVertice else 2
						point = C1
					elif (sqrLength2 <= sqrLength3):
						status = 3 if C2IsVertice else 2
						point = C2
					else:
						status = 3 if C3IsVertice else 2
						point = C3
	return status, point


@ti.func
def SpheresContactPoint(center1, center2, radius1, radius2):
	return center1 + (center2 - center1) * radius1 / (radius1 + radius2)


@ti.func
def GetBoundingBox(vertice1, vertice2, vertice3):
	coordBeg = vec3f(0, 0, 0)
	coordEnd = vec3f(0, 0, 0)

	coordBeg = ti.min(coordBeg, vertice1)
	coordBeg = ti.min(coordBeg, vertice2)
	coordBeg = ti.min(coordBeg, vertice3)

	coordEnd = ti.max(coordEnd, vertice1)
	coordEnd = ti.max(coordEnd, vertice2)
	coordEnd = ti.max(coordEnd, vertice3)
	return coordBeg, coordEnd

