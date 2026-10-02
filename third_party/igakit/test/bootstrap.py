#!/usr/bin/env python
import sys, os
import unittest

def getbuilddir():
    from distutils.util import get_platform
    plat_name = get_platform()
    x, y = sys.version_info[:2]
    buildlib = f"lib.{plat_name}-{x}.{y}"
    if hasattr(sys, 'pyston_version_info'):
        x, y = sys.pyston_version_info[:2]
        buildlib = f"{buildlib}-pyston{x}.{y}"
    if hasattr(sys, 'gettotalrefcount'):
        buildlib = f"{buildlib}-pydebug"
    return os.path.join("build", buildlib)

def bootstrap():
    try:
        import igakit
    except ImportError:
        from os.path import dirname, abspath, join
        top_dir = abspath(join(dirname(__file__), '..'))
        build_dir = join(top_dir, getbuilddir())
        sys.path.insert(0, build_dir)
        import igakit
        sys.stderr.write(
            "Loaded package 'igakit' from build/ directory\n")
    return igakit

bootstrap()
