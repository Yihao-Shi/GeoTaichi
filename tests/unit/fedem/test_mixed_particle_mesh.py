import numpy as np

from research.llm_assist.benchmarks.mixed_particle_mesh import (
    normalize_reference,
)
from src.fem.generator import FEMMesh


def test_reference_quality_reports_disconnected_volume_count():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)

    normalized, quality = normalize_reference(
        FEMMesh(points, cells, "TET4", name="two_disconnected_tetrahedra")
    )

    assert normalized.body_ids.tolist() == [0, 1]
    assert quality["connected_body_count"] == 2
