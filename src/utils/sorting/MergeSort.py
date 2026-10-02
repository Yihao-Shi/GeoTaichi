import taichi as ti

@ti.kernel
def merge_sort_key_value(length: int, keys: ti.template(), vals: ti.template(), temp_keys: ti.template(), temp_vals: ti.template()):
    width = 1
    stages = ti.ceil(ti.log(length, 2))
    for stage in range(stages):
        i = 0
        while i < length:
            left = i
            mid = ti.min(i + width, length)
            right = ti.min(i + 2 * width, length)
            l = left
            r = mid
            k = left
            while l < mid and r < right:
                if keys[l] <= keys[r]:
                    temp_keys[k] = keys[l]
                    temp_vals[k] = vals[l]
                    l += 1
                else:
                    temp_keys[k] = keys[r]
                    temp_vals[k] = vals[r]
                    r += 1
                k += 1
            while l < mid:
                temp_keys[k] = keys[l]
                temp_vals[k] = vals[l]
                l += 1
                k += 1
            while r < right:
                temp_keys[k] = keys[r]
                temp_vals[k] = vals[r]
                r += 1
                k += 1
            i += 2 * width
        for j in range(length):
            keys[j] = temp_keys[j]
            vals[j] = temp_vals[j]
        width *= 2