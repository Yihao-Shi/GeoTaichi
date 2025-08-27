import taichi as ti


def parallel_sort(keys, start, length):
    """Odd-even merge sort

    References:
        https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
        https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
    """
    num_stages = 0
    p = 1
    while p < length:
        k = p
        while k >= 1:
            invocations = int((length - k - k % p) / (2 * k)) + 1
            sort_stage1(keys, length, p, k, start, invocations)
            num_stages += 1
            ti.sync()
            k = int(k / 2)
        p = int(p * 2)
    
def parallel_sort_with_value(keys, values, start, length):
    """Odd-even merge sort

    References:
        https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
        https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
    """

    num_stages = 0
    p = 1
    while p < length:
        k = p
        while k >= 1:
            invocations = int((length - k - k % p) / (2 * k)) + 1
            sort_stage2(keys, values, length, p, k, start, invocations)
            num_stages += 1
            ti.sync()
            k = int(k / 2)
        p = int(p * 2)

def parallel_sort_struct_with_value(keys, values, start, length):
    """Odd-even merge sort

    References:
        https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
        https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
    """

    num_stages = 0
    p = 1
    while p < length:
        k = p
        while k >= 1:
            invocations = int((length - k - k % p) / (2 * k)) + 1
            sort_stage3(keys, values, length, p, k, start, invocations)
            num_stages += 1
            ti.sync()
            k = int(k / 2)
        p = int(p * 2)

@ti.kernel
def sort_stage1(keys: ti.template(), N: int, p: int, k: int, start: int, invocations: int):
    for inv in range(invocations):
        j = k % p + inv * 2 * k
        for i in range(0, ti.min(k, N - j - k)):
            a = i + j
            b = i + j + k
            if int(a / (p * 2)) == int(b / (p * 2)):
                key_a = keys[a+start]
                key_b = keys[b+start]
                if key_a > key_b:
                    keys[a+start] = key_b
                    keys[b+start] = key_a

@ti.kernel
def sort_stage2(keys: ti.template(), values: ti.template(), N: int, p: int, k: int, start: int, invocations: int):
    for inv in range(invocations):
        j = k % p + inv * 2 * k
        for i in range(0, ti.min(k, N - j - k)):
            a = i + j
            b = i + j + k
            if int(a / (p * 2)) == int(b / (p * 2)):
                key_a = keys[a+start]
                key_b = keys[b+start]
                if key_a > key_b:
                    keys[a+start] = key_b
                    keys[b+start] = key_a
                    values[a+start], values[b+start] = values[b+start], values[a+start]

@ti.kernel
def sort_stage3(keys: ti.template(), values: ti.template(), N: int, p: int, k: int, start: int, invocations: int):
    for inv in range(invocations):
        j = k % p + inv * 2 * k
        for i in range(0, ti.min(k, N - j - k)):
            a = i + j
            b = i + j + k
            if int(a / (p * 2)) == int(b / (p * 2)):
                key_a = keys[a+start]
                key_b = keys[b+start]
                if key_a > key_b:
                    values[a+start], values[b+start] = values[b+start], values[a+start]
