#!/bin/env python3

from distutils.core import setup, Extension

setup(name="custom",
      version="1.0",
      ext_modules=[Extension("lczero.backends", ["tmp.cc"])])