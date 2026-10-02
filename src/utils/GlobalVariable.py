
USEGPU = False
TRACKENERGY = False
CONSTANTORQUEMODEL = False
DIMENSION = 3

# Random field method
RANDOMFIELD = False

# lightweight MPM
BBAR = False
FBAR = False
TPIC = False
APIC = False
PARTICLESHIFTING = False
SHAPEFUNCTION = 0
INFLUENCENODE = 2
TWOPHASESINGLELAYER = False

# Constitutive model
DRIFTCORRECT = True

# PBC
MPMXPBC = False
MPMYPBC = False
MPMZPBC = False
MPMXSIZE = 0.
MPMYSIZE = 0.
MPMZSIZE = 0.
DEMXPBC = False
DEMYPBC = False
DEMZPBC = False
DEMXSIZE = 0.
DEMYSIZE = 0.
DEMZSIZE = 0.

# Nurbs calculation
max_degree = 2

# Contact model
ADAPTIVESTIFF = False
ENABLESHELL = False

# LSMPM soft--soft contact can optionally advance the normal penalty
# penetration from the same relative velocity used by the transposed surface
# force transfer.  The default keeps the legacy instantaneous-SDF path; paper
# diagnostics enable the work-conjugate path explicitly before Taichi kernels
# are compiled.
LSMPM_SOFT_SOFT_WORK_CONJUGATE = False
