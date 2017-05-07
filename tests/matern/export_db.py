#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mimclib
from mimclib import MIMCDatabase
import numpy as np
import sys

from subprocess import call
call(["tar", "-xvf", "/scratch/hajiali/Dropbox/Apps/terminal_up/mimc.tgz",
      '-C', '/scratch/hajiali/'])

selectTag = '%'
to_db = MIMCDatabase(engine="mysql", host='127.0.0.1', db='mimc')
from_db = MIMCDatabase(engine="sqlite", db='/scratch/hajiali/mimc.sqlite')
mimclib.db.export_db(selectTag, from_db, to_db)
