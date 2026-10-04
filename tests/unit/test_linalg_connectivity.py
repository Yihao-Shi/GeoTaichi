import numpy as np

from src.utils.linalg import remove_connectivity_by_inactive_faces


def test_remove_connectivity_accepts_scene_list_counts():
    connectivity, face_count, vertex_count = remove_connectivity_by_inactive_faces(
        np.asarray([[0, 1, 2], [3, 4, 5]]),
        [1, 1],
        [3, 3],
        np.asarray([False, True]),
    )

    np.testing.assert_array_equal(connectivity, [[0, 1, 2]])
    np.testing.assert_array_equal(face_count, [1])
    np.testing.assert_array_equal(vertex_count, [3])
