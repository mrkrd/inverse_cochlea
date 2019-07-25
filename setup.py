#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = "inverse_cochlea",
    version = "1",
    author = "Marek Rudnicki",
    author_email = "marek.rudnicki@tum.de",

    description = "Reconstruction of sounds from auditory nerve fibers' spikes",

    packages = find_packages(),
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Cython",
        "Programming Language :: C",
    ],

    platforms = ["Linux", "Windows", "FreeBSD", "OSX"],
    install_requires=["numpy", "pandas", "scipy", "ffnet", "joblib", "cochlea"],
)
