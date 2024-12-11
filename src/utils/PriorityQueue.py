import taichi as ti
from functools import reduce


@ti.data_oriented
class PriorityQueue:
    def __init__(self, dimension, resolution):
        self.max_length = reduce(lambda x, y : x * y, resolution)

        self.dist = ti.field(dtype=float)
        self.node = ti.Vector.field(dimension, dtype=ti.i32)
        ti.root.dense(ti.i, self.max_length).place(self.dist, self.node)
        self.total = ti.field(dtype=ti.i32, shape=())
    
    @ti.func
    def swap(self, i, j):
        tmp_dist, tmp_node = self.dist[i], self.node[i]
        self.dist[i], self.node[i] = self.dist[j], self.node[j]
        self.dist[j], self.node[j] = tmp_dist, tmp_node

    @ti.func
    def clear(self):
        self.total[None] = 0

    @ti.func
    def empty(self):
        return (self.total[None] == 0)

    @ti.func
    def top(self):
        assert(self.total[None] > 0)
        return self.node[1]

    @ti.func
    def pop(self):
        assert(self.total[None] > 0)
        self.swap(1, self.total[None])
        self.total[None].atomic_add(-1)
        
        now = 1
        now2 = now * 2
        nxt1, nxt2 = 0, 0

        while now2 <= self.total[None]:
            if now2 + 1 > self.total[None]:
                nxt1 = now2
                nxt2 = now2
            else:
                nxt1 = now2 if self.dist[now2] < self.dist[now2 + 1] else now2 + 1
                nxt2 = now2 if self.dist[now2] >= self.dist[now2 + 1] else now2 + 1

            if self.dist[nxt1] < self.dist[now]:
                self.swap(nxt1, now)
                now = nxt1
                now2 = now * 2
            elif self.dist[nxt2] < self.dist[now]:
                self.swap(nxt2, now)
                now = nxt2
                now2 = now * 2
            else:
                break

    @ti.func
    def push(self, dist, node):
        assert(self.total[None] < self.max_length)
        self.total[None].atomic_add(1)
        self.dist[self.total[None]] = dist
        self.node[self.total[None]] = node
        
        now = self.total[None]
        now_2 = now // 2
        while now >  1 and self.dist[now] < self.dist[now_2]:
            self.swap(now, now_2)
            now = now_2
            now_2 = now // 2
