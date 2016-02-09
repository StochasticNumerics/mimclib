import os
from setuptools import setup, find_packages, Extension


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="mimclib",
    version="0.0.1-dev",
    author="Abdul-Lateef Haji-Ali",
    author_email="abdullateef.hajiali@kaust.edu.sa",
    description="A library implementing the MIMC and CMLMC methods.",
    license="BSD",
    url="http://stochastic_numerics.kaust.edu.sa/",
    packages=find_packages(),
    long_description=read('README')
    ,ext_modules=[
        Extension('_set_util',
                  ['setutil/src/set_util.cpp', 'setutil/mimc/set_util.h',
                   'setutil/src/var_list.cpp', 'setutil/mimc/var_list.h'],
                  include_dirs=[''],
                  library_dirs=['/'],
                  libraries=[],
                  extra_compile_args=['-g']
                 )]
)
