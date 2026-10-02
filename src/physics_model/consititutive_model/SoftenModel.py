import taichi as ti


@ti.data_oriented
class LinearSoft:
    @ti.func
    def soft(self, parameter, peak_value, residual_value, start_value, end_value, current_value):
        value = peak_value
        if current_value > start_value:
            if current_value < end_value:
                value = peak_value - (peak_value - residual_value) * (current_value - start_value) / (end_value - start_value)
            else:
                value = residual_value
        return value

    @ti.func
    def soft_deriv(self, parameter, peak_value, residual_value, start_value, end_value, current_value):
        value = 0.
        if current_value > start_value:
            if current_value < end_value:
                value = (residual_value - peak_value) / (end_value - start_value)
        return value
    

@ti.data_oriented
class ExponentialSoft:
    @ti.func
    def soft(self, parameter, peak_value, residual_value, start_value, end_value, current_value):
        value = peak_value
        if current_value > start_value:
            value = residual_value + (peak_value - residual_value) * ti.exp(-parameter * (current_value - start_value))
        return value

    @ti.func
    def soft_deriv(self, parameter, peak_value, residual_value, start_value, end_value, current_value):
        value = 0.
        if current_value > start_value:
            value = -parameter * (peak_value - residual_value) * ti.exp(-parameter * (current_value - start_value))
        return value
    

@ti.data_oriented
class SinhSoft:
    @ti.func
    def sinh(self, value):
        return 0.5 * (ti.exp(value) - ti.exp(-value))
    
    @ti.func
    def dsinh(self, value):
        return 0.5 * (ti.exp(value) + ti.exp(-value))

    @ti.func
    def soft(self, parameter, peak_value, residual_value, start_value, end_value, current_value):
        value = peak_value * self.sinh(-parameter * current_value)
        return value

    @ti.func
    def soft_deriv(self, parameter, peak_value, residual_value, start_value, end_value, current_value):
        value = -parameter * peak_value * self.dsinh(-parameter * current_value)
        return value