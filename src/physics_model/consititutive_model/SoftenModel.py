import taichi as ti


@ti.data_oriented
class LinearSoft:
    def __init__(self, parameter):
        self.parameter = parameter

    @ti.func
    def soft(self, peak_value, residual_value, start_value, end_value, current_value):
        value = peak_value
        if current_value > start_value:
            if current_value < end_value:
                value = peak_value - self.parameter * (peak_value - residual_value) * (current_value - start_value) / (end_value - start_value)
            else:
                value = residual_value
        return value

    @ti.func
    def soft_deriv(self, peak_value, residual_value, start_value, end_value, current_value):
        value = 0.
        if current_value > start_value:
            if current_value < end_value:
                value = self.parameter * (residual_value - peak_value) / (end_value - start_value)
        return value
    

@ti.data_oriented
class ExponentialSoft:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    @ti.func
    def soft(self, peak_value, residual_value, start_value, end_value, current_value):
        value = peak_value
        if current_value > start_value:
            value = residual_value + (peak_value - residual_value) * self.alpha * ti.exp(-self.beta * (current_value - start_value))
        return value

    @ti.func
    def soft_deriv(self, peak_value, residual_value, start_value, end_value, current_value):
        value = 0.
        if current_value > start_value:
            value = -self.beta * (peak_value - residual_value) * self.alpha * ti.exp(-self.beta * (current_value - start_value))
        return value
    

@ti.data_oriented
class SinhSoft:
    def __init__(self, parameter):
        self.parameter = parameter

    @ti.func
    def sinh(self, value):
        return 0.5 * (ti.exp(value) - ti.exp(-value))
    
    @ti.func
    def dsinh(self, value):
        return 0.5 * (ti.exp(value) + ti.exp(-value))

    @ti.func
    def soft(self, peak_value, residual_value, start_value, end_value, current_value):
        value = peak_value * self.sinh(-self.parameter * current_value)
        return value

    @ti.func
    def soft_deriv(self, peak_value, residual_value, start_value, end_value, current_value):
        value = -self.parameter * peak_value * self.dsinh(-self.parameter * current_value)
        return value
