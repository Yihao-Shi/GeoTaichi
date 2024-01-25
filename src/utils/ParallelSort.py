import taichi as ti


class Sort(object):
    def __init__(self) -> None:
        pass

    def parallel_sort(self, keys):
        """Odd-even merge sort

        References:
            https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
            https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
        """
        N = keys.shape[0]

        num_stages = 0
        p = 1
        while p < N:
            k = p
            while k >= 1:
                invocations = int((N - k - k % p) / (2 * k)) + 1
                sort_stage1(keys, N, p, k, invocations)
                num_stages += 1
                ti.sync()
                k = int(k / 2)
            p = int(p * 2)
        
    def parallel_sort_with_value(self, keys, values):
        """Odd-even merge sort

        References:
            https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
            https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
        """
        N = keys.shape[0]

        num_stages = 0
        p = 1
        while p < N:
            k = p
            while k >= 1:
                invocations = int((N - k - k % p) / (2 * k)) + 1
                sort_stage2(keys, 1, values, N, p, k, invocations)
                num_stages += 1
                ti.sync()
                k = int(k / 2)
            p = int(p * 2)

@ti.kernel
def sort_stage1(keys: ti.template(), N: int, p: int, k: int, invocations: int):
    for inv in range(invocations):
        j = k % p + inv * 2 * k
        for i in range(0, ti.min(k, N - j - k)):
            a = i + j
            b = i + j + k
            if int(a / (p * 2)) == int(b / (p * 2)):
                key_a = keys[a]
                key_b = keys[b]
                if key_a > key_b:
                    keys[a] = key_b
                    keys[b] = key_a

@ti.kernel
def sort_stage2(keys: ti.template(), values: ti.template(), N: int, p: int, k: int, invocations: int):
    for inv in range(invocations):
        j = k % p + inv * 2 * k
        for i in range(0, ti.min(k, N - j - k)):
            a = i + j
            b = i + j + k
            if int(a / (p * 2)) == int(b / (p * 2)):
                key_a = keys[a]
                key_b = keys[b]
                if key_a > key_b:
                    keys[a] = key_b
                    keys[b] = key_a
                    temp = values[a]
                    values[a] = values[b]
                    values[b] = temp