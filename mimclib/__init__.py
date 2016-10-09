from __future__ import absolute_import

from .mimc import MIMCRun, MIMCItrData
from . import plot
from .db import MIMCDatabase


try:
    import pkg_resources  # part of setuptools
    __version__ = pkg_resources.require("mimclib")[0].version
except:
    __version__ = "Not installed"
