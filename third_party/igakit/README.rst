igakit: Toolkit for IsoGeometric Analysis (IGA)
===============================================


Overview
--------

This package implements many of the NURBS routines in Piegl's book
using Fortran. It provides Python bindings for these functions using
``f2py``. Finally, we provide a NURBS class which uses this
functionality for the simplified, manually creation of geometries for
use in isogeometric analysis.


Installation
------------

Quick installation::

  $ pip install https://github.com/dalcinl/igakit/archive/refs/heads/master.zip

Alternatively, clone the repository hosted at
`GitHub <https://github.com/dalcinl/igakit>`_::

  $ git clone https://github.com/dalcinl/igakit

and next install from the top level package directory using ``pip``::

  $ cd igakit
  $ python -m pip install .


Acknowledgments
---------------

This project was partially supported by the
Advanced Algorithm and Numerical Simulations Laboratory
(`AANSLab <https://cemse.kaust.edu.sa/aanslab>`_),
Division of Computer, Electrical, and
Mathematical Sciences & Engineering
(`CEMSE <https://cemse.kaust.edu.sa>`_),
King Abdullah University of Science and Technology
(`KAUST <http://www.kaust.edu.sa>`_).
