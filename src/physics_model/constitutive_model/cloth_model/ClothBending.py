"""Local quadratic and signed-dihedral cloth bending laws."""

import taichi as ti


@ti.data_oriented
class ClothBending:
    @staticmethod
    def validate_model(value):
        if not isinstance(value, str) or value not in ("None", "Quadratic", "Dihedral"):
            raise ValueError("cloth bending_model must be 'None', 'Quadratic', or 'Dihedral'")
        return value

    @staticmethod
    def quadratic_modulus(stiffness, thickness, poisson_ratio):
        return stiffness * thickness**3 / (24.0 * (1.0 - poisson_ratio**2))

    @ti.func
    def _hinge_projection_direction(self, point, edge0, edge1):
        edge = edge1 - edge0
        denominator = ti.max(edge.dot(edge), 1.0e-30)
        direction = edge0 + (point - edge0).dot(edge) / denominator * edge - point
        return direction / ti.max(direction.norm(), 1.0e-15)

    @ti.func
    def signed_dihedral_angle(self, x0, x1, x2, x3):
        normal0 = (x1 - x0).cross(x2 - x0)
        normal1 = (x2 - x3).cross(x1 - x3)
        denominator = ti.max(normal0.norm() * normal1.norm(), 1.0e-30)
        cosine = ti.max(-1.0, ti.min(1.0, normal0.dot(normal1) / denominator))
        angle = ti.acos(cosine)
        if normal1.cross(normal0).dot(x1 - x2) < 0.0:
            angle = -angle
        return angle

    @ti.func
    def signed_dihedral_gradient(self, x0, x1, x2, x3):
        edge0 = x2 - x1
        edge1 = x0 - x1
        edge2 = x3 - x1
        edge3 = x0 - x2
        edge4 = x3 - x2
        normal0 = edge0.cross(edge1)
        normal1 = edge2.cross(edge0)
        normal0_squared = ti.max(normal0.dot(normal0), 1.0e-30)
        normal1_squared = ti.max(normal1.dot(normal1), 1.0e-30)
        hinge_length = ti.max(edge0.norm(), 1.0e-15)
        gradient = ti.Vector.zero(float, 12)
        value0 = -hinge_length / normal0_squared * normal0
        value1 = (
            -edge0.dot(edge3) / (hinge_length * normal0_squared) * normal0
            - edge0.dot(edge4) / (hinge_length * normal1_squared) * normal1
        )
        value2 = (
            edge0.dot(edge1) / (hinge_length * normal0_squared) * normal0
            + edge0.dot(edge2) / (hinge_length * normal1_squared) * normal1
        )
        value3 = -hinge_length / normal1_squared * normal1
        for component in ti.static(range(3)):
            gradient[component] = value0[component]
            gradient[3 + component] = value1[component]
            gradient[6 + component] = value2[component]
            gradient[9 + component] = value3[component]
        return gradient

    @ti.func
    def signed_dihedral_hessian(self, x0, x1, x2, x3):
        edge0 = x2 - x1
        edge1 = x0 - x1
        edge2 = x3 - x1
        edge3 = x0 - x2
        edge4 = x3 - x2
        length0 = ti.max(edge0.norm(), 1.0e-15)
        length1 = ti.max(edge1.norm(), 1.0e-15)
        length2 = ti.max(edge2.norm(), 1.0e-15)
        length3 = ti.max(edge3.norm(), 1.0e-15)
        length4 = ti.max(edge4.norm(), 1.0e-15)
        normal0 = edge0.cross(edge1)
        normal1 = edge2.cross(edge0)
        normal0_length = ti.max(normal0.norm(), 1.0e-15)
        normal1_length = ti.max(normal1.norm(), 1.0e-15)

        m1 = self._hinge_projection_direction(x2, x1, x0)
        m2 = self._hinge_projection_direction(x2, x1, x3)
        m3 = self._hinge_projection_direction(x1, x2, x0)
        m4 = self._hinge_projection_direction(x1, x2, x3)
        m01 = self._hinge_projection_direction(x0, x1, x2)
        m02 = self._hinge_projection_direction(x3, x1, x2)
        cosine1 = edge0.dot(edge1) / (length0 * length1)
        cosine2 = edge0.dot(edge2) / (length0 * length2)
        cosine3 = -edge0.dot(edge3) / (length0 * length3)
        cosine4 = -edge0.dot(edge4) / (length0 * length4)
        height1 = normal0_length / length1
        height2 = normal1_length / length2
        height3 = normal0_length / length3
        height4 = normal1_length / length4
        height01 = normal0_length / length0
        height02 = normal1_length / length0

        n1_01 = normal0.outer_product(m01) / (height01 * height01 * normal0_length)
        n2_02 = normal1.outer_product(m02) / (height02 * height02 * normal1_length)
        n1_3 = normal0.outer_product(m3) / (height01 * height3 * normal0_length)
        n1_1 = normal0.outer_product(m1) / (height01 * height1 * normal0_length)
        n2_4 = normal1.outer_product(m4) / (height02 * height4 * normal1_length)
        n2_2 = normal1.outer_product(m2) / (height02 * height2 * normal1_length)
        m3_01_1 = cosine3 / (height3 * height01 * normal0_length) * m01.outer_product(normal0)
        m1_01_1 = cosine1 / (height1 * height01 * normal0_length) * m01.outer_product(normal0)
        m1_1_1 = cosine1 / (height1 * height1 * normal0_length) * m1.outer_product(normal0)
        m3_3_1 = cosine3 / (height3 * height3 * normal0_length) * m3.outer_product(normal0)
        m3_1_1 = cosine3 / (height3 * height1 * normal0_length) * m1.outer_product(normal0)
        m1_3_1 = cosine1 / (height1 * height3 * normal0_length) * m3.outer_product(normal0)
        m4_02_2 = cosine4 / (height4 * height02 * normal1_length) * m02.outer_product(normal1)
        m2_02_2 = cosine2 / (height2 * height02 * normal1_length) * m02.outer_product(normal1)
        m4_4_2 = cosine4 / (height4 * height4 * normal1_length) * m4.outer_product(normal1)
        m2_4_2 = cosine2 / (height2 * height4 * normal1_length) * m4.outer_product(normal1)
        m4_2_2 = cosine4 / (height4 * height2 * normal1_length) * m2.outer_product(normal1)
        m2_2_2 = cosine2 / (height2 * height2 * normal1_length) * m2.outer_product(normal1)
        b1 = normal0.outer_product(m01) / (length0 * length0 * normal0_length)
        b2 = normal1.outer_product(m02) / (length0 * length0 * normal1_length)

        block00 = -(n1_01 + n1_01.transpose())
        block10 = m3_01_1 - n1_3
        block20 = m1_01_1 - n1_1
        block11 = m3_3_1 + m3_3_1.transpose() - b1 + m4_4_2 + m4_4_2.transpose() - b2
        block12 = m3_1_1 + m1_3_1.transpose() + b1 + m4_2_2 + m2_4_2.transpose() + b2
        block13 = m4_02_2 - n2_4
        block22 = m1_1_1 + m1_1_1.transpose() - b1 + m2_2_2 + m2_2_2.transpose() - b2
        block23 = m2_02_2 - n2_2
        block33 = -(n2_02 + n2_02.transpose())
        angle_hessian = ti.Matrix.zero(float, 12, 12)
        for row, column in ti.static(ti.ndrange(3, 3)):
            angle_hessian[row, column] = block00[row, column]
            angle_hessian[3 + row, column] = block10[row, column]
            angle_hessian[row, 3 + column] = block10[column, row]
            angle_hessian[6 + row, column] = block20[row, column]
            angle_hessian[row, 6 + column] = block20[column, row]
            angle_hessian[3 + row, 3 + column] = block11[row, column]
            angle_hessian[3 + row, 6 + column] = block12[row, column]
            angle_hessian[6 + row, 3 + column] = block12[column, row]
            angle_hessian[3 + row, 9 + column] = block13[row, column]
            angle_hessian[9 + row, 3 + column] = block13[column, row]
            angle_hessian[6 + row, 6 + column] = block22[row, column]
            angle_hessian[6 + row, 9 + column] = block23[row, column]
            angle_hessian[9 + row, 6 + column] = block23[column, row]
            angle_hessian[9 + row, 9 + column] = block33[row, column]
        angle_hessian = 0.5 * (angle_hessian + angle_hessian.transpose())
        return angle_hessian

    @ti.func
    def quadratic_response(self, stiffness, x0, x1, x2, x3):
        points = ti.Matrix.rows([x0, x1, x2, x3])
        value = 0.0
        gradient = ti.Vector.zero(float, 12)
        for first, second in ti.static(ti.ndrange(4, 4)):
            value += 0.5 * stiffness[first, second] * points[first, :].dot(points[second, :])
            for component in ti.static(range(3)):
                gradient[3 * first + component] += stiffness[first, second] * points[second, component]
        return value, gradient

    @ti.func
    def dihedral_response(self, modulus, edge_length, height, rest_angle, x0, x1, x2, x3):
        angle = self.signed_dihedral_angle(x0, x1, x2, x3)
        angle_gradient = self.signed_dihedral_gradient(x0, x1, x2, x3)
        difference = angle - rest_angle
        ratio = edge_length / height
        value = modulus * ratio * difference * difference
        gradient = 2.0 * modulus * ratio * difference * angle_gradient
        return value, gradient

    @ti.func
    def dihedral_tangent(self, modulus, edge_length, height, rest_angle, x0, x1, x2, x3, project_pd: ti.template()):
        angle = self.signed_dihedral_angle(x0, x1, x2, x3)
        angle_gradient = self.signed_dihedral_gradient(x0, x1, x2, x3)
        difference = angle - rest_angle
        ratio = edge_length / height
        second = 2.0 * modulus * ratio
        tangent = second * angle_gradient.outer_product(angle_gradient)
        if ti.static(not project_pd):
            angle_hessian = self.signed_dihedral_hessian(x0, x1, x2, x3)
            tangent = 2.0 * modulus * ratio * difference * angle_hessian + tangent
            tangent = 0.5 * (tangent + tangent.transpose())
        # The projected path uses the PSD Gauss--Newton block, avoiding a 12x12 eigensolve.
        return tangent
