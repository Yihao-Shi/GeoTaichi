import taichi as ti


@ti.data_oriented
class HASHMAP:
    def __init__(self, HashSize, TensorSize, type):
        self.HashSize = HashSize
        if type == 0:                                                                                    # Scalar Hash Map
            self.map = ti.field(float, HashSize)
        elif type == 1:                                                                                  # Vector Hash Map
            self.map = ti.Vector.field(TensorSize, float, HashSize)       
        elif type == 2:                                                                                  # Matrix Hash Map
            self.map = ti.Matrix.field(TensorSize, TensorSize, float, HashSize)
        self.Key = ti.field(ti.u64, HashSize)
    
    @ti.func
    def get_hash(self, key: ti.u64):
        return int(key % self.HashSize)

    @ti.func
    def rehash(self, slot, attempt):
        radix = (attempt + 1) // 2 
        return int((slot + (-1) ** (attempt - 1) * radix * radix) % self.HashSize)

    @ti.func
    def MapInit(self, key):
        self.Key[key] = 0

    @ti.func
    def isConflict(self):
        pass

    @ti.func
    def Insert(self, key: ti.u64, value):
        slot = self.get_hash(key)
        attempt = 1
        while self.Key[slot] != 0:
            slot = self.rehash(slot, attempt)
            attempt += 1
        self.Key[slot] = key
        self.map[slot] = value
        

    @ti.func
    def Search(self, key: ti.u64):
        slot = self.get_hash(key)
        attempt = 1
        while self.Key[slot] != 0:
            if self.Key[slot] == key:
                break
            slot = self.rehash(slot, attempt)
            attempt += 1
        if self.Key[slot] == 0: slot = -1
        return slot

    @ti.func
    def Delete(self):
        pass

