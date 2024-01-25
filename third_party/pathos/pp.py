#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
#
"""
``pathos`` interface to the ``pp`` (parallel python) module.

Notes:
    This module has been deprecated in favor of ``pathos.parallel``.
"""
# backward compatibility
__all__ = ['ParallelPythonPool', 'stats']
from third_party.pathos.parallel import __doc__, __print_stats, __STATE
from third_party.pathos.parallel import *
ParallelPythonPool = ParallelPool

# EOF
