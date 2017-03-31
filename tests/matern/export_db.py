#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mimclib import MIMCDatabase
import numpy as np
import sys

selectTag = 'matern-new-miproj%'
to_db = MIMCDatabase(engine="mysql", host='127.0.0.1', db='mimc')
from_db = MIMCDatabase(engine="sqlite", db='mimc.sqlite')

with from_db.DBConn(**from_db.connArgs) as from_cur:
    with to_db.DBConn(**to_db.connArgs) as to_cur:
        print("Getting runs")
        runs = np.array(from_cur.execute(
            'SELECT run_id, creation_date, TOL, done_flag, tag, totalTime, comment, fn, params FROM tbl_runs WHERE tag LIKE ?',
            [selectTag]).fetchall())
        for i, r in enumerate(runs):
            to_cur.execute('INSERT INTO tbl_runs(creation_date, TOL, done_flag, tag, totalTime, comment, fn, params)\
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)', r[1:])
            new_run_id = to_cur.getLastRowID()

            iters = np.array(from_cur.execute(
                'SELECT iter_id, TOL, bias, stat_error, creation_date, totalTime, Qparams, userdata, iteration_idx, exact_error FROM tbl_iters WHERE run_id=?',
                [r[0]]).fetchall())
            for j, itr in enumerate(iters):
                sys.stdout.write("\rDoing itr {}/{} {}/{}".format(i, len(runs), j, len(iters)))
                sys.stdout.flush()
                to_cur.execute('INSERT INTO tbl_iters(run_id, TOL, bias, stat_error, creation_date, totalTime, Qparams, userdata, iteration_idx, exact_error)\
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [new_run_id] + itr[1:].tolist())
                new_iter_id = to_cur.getLastRowID()

                lvls = np.array(from_cur.execute(
                    'SELECT lvl, lvl_hash, El, Vl, Wl, tT, tW, Ml, psums_delta, psums_fine FROM tbl_lvls WHERE iter_id=?',
                    [itr[0]]).fetchall())
                for lvl in lvls:
                    to_cur.execute('INSERT INTO tbl_lvls(iter_id, lvl, lvl_hash, El, Vl, Wl, tT, tW, Ml, psums_delta, psums_fine)\
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [new_iter_id] + lvl.tolist())
            sys.stdout.write('\n')
