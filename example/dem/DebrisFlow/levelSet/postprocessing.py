from geotaichi import *

init(arch='gpu', log=False, debug=False, device_memory_GB=4)

lsdem = DEM()

lsdem.postprocessing(start_file=10,end_file=11, scheme="LSDEM", read_path="OutputData")
