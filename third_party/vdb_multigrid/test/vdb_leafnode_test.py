import sys
sys.path.append("../src/")
import unittest
import taichi as ti
from src.vdb_grid import *

ti.init(arch=ti.cuda, device_memory_GB=4)


class VdbLeafNodeTest(unittest.TestCase):

    @staticmethod
    def test_insert_node():
        # Test Direct Insert Root
        s0 = ti.root
        VdbLeafNode.append_leaf_node(s0, 2, 2, 2)

        # Test Insert Intermediate Node
        s1 = s0.pointer(ti.i, 16)
        VdbLeafNode.append_leaf_node(s1, 4, 4, 4)


if __name__=='__main__':
    unittest.main()
