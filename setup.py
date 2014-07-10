#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = "inverse_cochlea",
    version = "0.1",
    author = "Marek Rudnicki",
    author_email = "marek.rudnicki@tum.de",

    description = "Reconstruction of sounds from auditory nerve fibers' spikes",

    packages = find_packages(),
)
