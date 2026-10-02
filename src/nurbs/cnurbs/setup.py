from third_party.pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

__version__ = "0.0.1"
cxx_std=11

ext_modules = \
[
    Pybind11Extension("nurbs_one_basis",
        ["NURBSOneBasis.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args= ['-g', '-O3']
        ),
        
    Pybind11Extension("nurbs_interpolation",
        ["NURBSinterpolation.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args= ['-g', '-O3']
        ),

    Pybind11Extension("nurbs_basis_ders",
        ["NURBSbasisDers.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args= ['-g', '-O3']
        ),

    Pybind11Extension("hrefine_nurbs",
        ["hRefineNURBS.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args= ['-g', '-O3']
        )
]

setup(
    name="nurbs",
    version=__version__,
    author="Yihao Shi",
    description="NURBS shape function and its deverivate via C++ backend",
    include_dirs="/usr/include/eigen3/",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    zip_safe=False,
    python_requires=">=3.7",
)
