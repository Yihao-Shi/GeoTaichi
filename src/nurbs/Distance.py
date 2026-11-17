import heapq


class Queue:
    def __init__(self):
        self.queue = []

    def append(self, patchA, patchB):
        width = max(
            patchA[0][1] - patchA[0][0],  # u-range of A
            patchA[1][1] - patchA[1][0],  # v-range of A
            patchB[0][1] - patchB[0][0],  # u-range of B
            patchB[1][1] - patchB[1][0],  # v-range of B
        )
        heapq.heappush(self.queue, (width, patchA, patchB))

    def pop(self):
        width, patch_a, patch_b = heapq.heappop(self.queue)
        return patch_a, patch_b

    def size(self):
        return len(self.queue)
