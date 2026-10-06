import sys
from pathlib import Path

sys.path.insert(0, "/Volumes/MyWork/GeoTaichi/research/remote_validation/cpt_ipc_20261004")
from pull_cpt_first10_when_done import collect

collect(
    "cpt_iga_volume_consistent_first10_20261006",
    Path("/Volumes/MyWork/GeoTaichi/examples/igampm/cpt_dp/OutputData/cpt_ipc_volume_consistent_first10_20261006"),
    True,
)
