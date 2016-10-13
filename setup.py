import os
from setuptools import setup, find_packages, Extension

from setuptools.command.install import install as cmd_install
class install(cmd_install, object):
    def run(self): super(install, self).run()

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

setup(
    name="mimclib",
    # 1st is increased if it is not compatible with previous version.
    # 2nd is increased if new features are added.
    # 3rd is increased for small fixes.
    version="1.0.2.dev4",
    author="Abdul-Lateef Haji-Ali",
    author_email="abdullateef.hajiali@kaust.edu.sa",
    description="A library implementing the MIMC and CMLMC methods.",
    license="BSD",
    url="http://stochastic_numerics.kaust.edu.sa/",
    packages=find_packages(),
    long_description=read('README.md'),
    install_requires=[],#['matplotlib>=1.5', 'numpy>=1.9', 'scipy>=0.17.0', 'dill'],
    extras_require={'mysqldb':  ["MySQL-python"]},
    ext_modules=[
        Extension('mimclib.libset_util',
                  ['mimclib/libsetutil/src/set_util.cpp',
                   'mimclib/libsetutil/src/var_list.cpp',],
                  include_dirs=[''],
                  library_dirs=['/'],
                  libraries=[],
                  extra_compile_args=['-std=c++11'])],
    cmdclass={'install': install},
)


# Need to install the package: libmysqlclient-dev
