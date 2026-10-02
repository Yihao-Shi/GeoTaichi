"""Compatibility shim for tools that still invoke ``setup.py`` directly.

Project metadata, dependencies, package discovery, data files, and console
entry points are owned by ``pyproject.toml``.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
