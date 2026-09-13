"""Compatibility shim for legacy pip versions without PEP 660 support."""

from setuptools import find_packages, setup


setup(
    name="cnsf-neural-tracking",
    version="0.1.0",
    description="Track-MT3, capacity-matched Track-MT3, and CNSF tracking research code",
    packages=find_packages(include=("track_mt3", "track_mt3.*")),
    python_requires=">=3.9",
    install_requires=["numpy>=1.24", "PyYAML>=6", "scipy>=1.10", "torch>=2.1"],
)
