from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cPickle
from . import setutil

import hashlib
__all__ = []

def public(sym):
    __all__.append(sym.__name__)
    return sym


def _md5(string):
    return hashlib.md5(string).hexdigest()

def _pickle(obj, dump=cPickle.dump):
    import io
    import MySQLdb
    with io.BytesIO() as f:
        dump(obj, f, protocol=2)
        f.seek(0)
        return MySQLdb.Binary(f.read())


def _unpickle(obj, load=cPickle.load):
    import io
    with io.BytesIO(obj) as f:
        return load(f)

def _nan2none(arr):
    return [None if np.isnan(x) else x for x in arr]

def _none2nan(x):
    return np.nan if x is None else x

class MySQLDBConn(object):
    def __init__(self, **kwargs):
        self.connArgs = kwargs

    def __enter__(self):
        import MySQLdb
        self.conn = MySQLdb.connect(compress=True, **self.connArgs)
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, type, value, traceback):
        self.Commit()
        self.conn.close()

    def execute(self, query, params=[]):
        query = query.replace("datetime()", "now()")
        query = query.replace("?", "%s")
        self.cur.execute(query, tuple(params))
        return self.cur

    def getLastRowID(self):
        return self.cur.lastrowid

    def getRowCount(self):
        return self.cur.rowcount

    def Commit(self):
        self.conn.commit()

    @staticmethod
    def DBCreationScript(drop_db=False, db="mimc"):
        script = ""
        if drop_db:
            script += "DROP DATABASE IF EXISTS {DBName};".format(DBName=db)
        script += '''
CREATE DATABASE IF NOT EXISTS {DBName};
USE {DBName};
CREATE TABLE IF NOT EXISTS tbl_runs (
    run_id                INTEGER PRIMARY KEY AUTO_INCREMENT NOT NULL,
    creation_date           DATETIME NOT NULL,
    TOL                   REAL NOT NULL,
    done_flag            INTEGER NOT NULL,
    totalTime               REAL,
    tag                   VARCHAR(128) NOT NULL,
    params                mediumblob,
    fn                    mediumblob,
    comment               TEXT
);
CREATE VIEW vw_runs AS SELECT run_id, creation_date, TOL, done_flag, tag, totalTime, comment FROM tbl_runs;

CREATE TABLE IF NOT EXISTS tbl_iters (
    iter_id                 INTEGER PRIMARY KEY AUTO_INCREMENT NOT NULL,
    run_id                  INTEGER NOT NULL,
    TOL                     REAL,
    bias                    REAL,
    stat_error              REAL,
    creation_date           DATETIME NOT NULL,
    totalTime               REAL,
    Qparams                 mediumblob,
    userdata                mediumblob,
    iteration_idx           INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES tbl_runs(run_id) ON DELETE CASCADE,
    UNIQUE KEY idx_itr_idx (run_id, iteration_idx)
);
CREATE VIEW vw_iters AS SELECT iter_id, run_id, TOL,
creation_date, bias, stat_error, totalTime, iteration_idx FROM tbl_iters;

CREATE TABLE IF NOT EXISTS tbl_lvls (
    iter_id       INTEGER NOT NULL,
    lvl           text NOT NULL,
    lvl_hash      varchar(35) NOT NULL,
    El            REAL,
    Vl            REAL,
    Wl            REAL,
    tT            REAL,
    tW            REAL,
    Ml            BIGINT,
    psums_delta   mediumblob,
    psums_fine    mediumblob,
    FOREIGN KEY (iter_id) REFERENCES tbl_iters(iter_id) ON DELETE CASCADE,
    UNIQUE KEY idx_run_lvl (iter_id, lvl_hash)
);

CREATE VIEW vw_lvls AS SELECT iter_id, lvl, El, Vl, Wl, tT, tW, Ml FROM tbl_lvls;

-- CREATE USER 'USER'@'%';
-- GRANT ALL PRIVILEGES ON *.* TO 'USER'@'%' WITH GRANT OPTION;
'''.format(DBName=db)
        return script


class SQLiteDBConn(object):
    def __init__(self, **kwargs):
        if "db" in kwargs:
            kwargs["database"] = kwargs.pop("db")
        self.connArgs = kwargs
        if "database" in kwargs:
            import os.path
            if not os.path.isfile(kwargs.get("database")):
                with self:
                    self.execute(SQLiteDBConn.DBCreationScript())
        with self:
            self.execute("PRAGMA foreign_keys = ON;")

    def __enter__(self):
        import sqlite3
        self.conn = sqlite3.connect(**self.connArgs)
        self.conn.text_factory = str
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, type, value, traceback):
        self.Commit()
        self.conn.close()

    def execute(self, query, params=[]):
        if len(params) > 0 and len(query.split(';')) > 1:
            raise Exception("Multiple queries with parameters is unsupported")

        # Expand lists in paramters
        prev = -1
        new_params = []
        for p in params:
            prev = query.find('?', prev+1)
            if type(p) in [np.uint16, np.uint32, np.uint64]:
                new_params.append(np.int64(p))  # sqlite is really fussy about this
            elif type(p) in [list, tuple]:
                rep = "(" + ",".join("?"*len(p)) + ")"
                query = query[:prev] + rep + query[prev+1:]
                prev += len(rep)
                new_params.extend(p)
            else:
                new_params.append(p)

        for q in query.split(';'):
            self.cur.execute(q, tuple(new_params))
        return self.cur

    def getLastRowID(self):
        return self.cur.lastrowid

    def getRowCount(self):
        return self.cur.rowcount

    def Commit(self):
        self.conn.commit()

    @staticmethod
    def DBCreationScript():
        script = '''
CREATE TABLE IF NOT EXISTS tbl_runs (
    run_id                INTEGER PRIMARY KEY NOT NULL,
    creation_date           DATETIME NOT NULL,
    TOL                   REAL NOT NULL,
    done_flag            INTEGER NOT NULL,
    totalTime               REAL,
    tag                   VARCHAR(128) NOT NULL,
    params                mediumblob,
    fn                    mediumblob,
    comment               TEXT
);
CREATE VIEW vw_runs AS SELECT run_id, creation_date, TOL, done_flag, tag, totalTime, comment FROM tbl_runs;

CREATE TABLE IF NOT EXISTS tbl_iters (
    iter_id                 INTEGER PRIMARY KEY NOT NULL,
    run_id                  INTEGER NOT NULL,
    TOL                     REAL,
    bias                    REAL,
    stat_error              REAL,
    creation_date           DATETIME NOT NULL,
    totalTime               REAL,
    Qparams                 mediumblob,
    userdata                mediumblob,
    iteration_idx           INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES tbl_runs(run_id) ON DELETE CASCADE,
    CONSTRAINT idx_itr_idx UNIQUE (run_id, iteration_idx)
);
CREATE VIEW vw_iters AS SELECT iter_id, run_id, TOL,
creation_date, bias, stat_error, totalTime, iteration_idx FROM tbl_iters;

CREATE TABLE IF NOT EXISTS tbl_lvls (
    iter_id       INTEGER NOT NULL,
    lvl           text NOT NULL,
    lvl_hash      varchar(35) NOT NULL,
    El            REAL,
    Vl            REAL,
    Wl            REAL,
    tT            REAL,
    tW            REAL,
    Ml            INTEGER,
    psums_delta   mediumblob,
    psums_fine    mediumblob,
    FOREIGN KEY (iter_id) REFERENCES tbl_iters(iter_id) ON DELETE CASCADE,
    CONSTRAINT idx_run_lvl UNIQUE (iter_id, lvl_hash)
);

CREATE VIEW vw_lvls AS SELECT iter_id, lvl, El, Vl, Wl, tT, tW, Ml FROM tbl_lvls;
'''
        return script

@public
class MIMCDatabase(object):
    def __init__(self, engine='mysql', **kwargs):
        self.DBName = kwargs.pop("db", 'mimc')
        kwargs["db"] = self.DBName
        self.engine = engine
        if self.engine == "mysql":
            self.DBConn = MySQLDBConn
        elif self.engine == 'sqlite':
            self.DBConn = SQLiteDBConn
        else:
            raise Exception("Unrecognized DB engine")

        self.connArgs = kwargs.copy()

    def createRun(self, tag, TOL=None, params=None, fn=None,
                  mimc_run=None, comment=""):
        TOL = TOL or mimc_run.params.TOL
        params = params or mimc_run.params
        fn = fn or dict(filter(lambda i:i[0] in "Norm",
                               mimc_run.fn.getDict().iteritems())) # Only save the Norm function
        import dill
        with self.DBConn(**self.connArgs) as cur:
            cur.execute('''
            INSERT INTO tbl_runs(creation_date, TOL, tag, params, fn, done_flag, comment)
            VALUES(datetime(), ?, ?, ?, ?, -1, ?)''',
                        [TOL, tag, _pickle(params), _pickle(fn, dump=dill.dump), comment])
            return cur.getLastRowID()

    def markRunDone(self, run_id, flag, totalTime=None, comment=''):
        with self.DBConn(**self.connArgs) as cur:
            cur.execute('''UPDATE tbl_runs SET done_flag=?, totalTime=?,
            comment = {}
            WHERE run_id=?'''.format('CONCAT(comment,  ?)' if self.engine=='mysql' else
            'comment || ?'), [flag, totalTime, comment, run_id])

    def markRunSuccessful(self, run_id, totalTime=None, comment=''):
        self.markRunDone(run_id, flag=1, comment=comment, totalTime=totalTime)

    def markRunFailed(self, run_id, totalTime=None, comment='', add_exception=True):
        if add_exception:
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            if exc_obj is not None:
                comment += "{}: {}".format(exc_type.__name__, exc_obj)
        self.markRunDone(run_id, flag=0, comment=comment, totalTime=totalTime)

    def writeRunData(self, run_id, mimc_run, iteration_idx, userdata=None):
        base = 0
        iteration = mimc_run.iters[iteration_idx]
        El = mimc_run.fn.Norm(iteration.calcDeltaEl())
        Vl = iteration.Vl_estimate
        tT = iteration.tT
        tW = iteration.tW
        Wl = iteration.Wl_estimate
        Ml = iteration.M

        prev_iter = mimc_run.iters[iteration_idx-1] if iteration_idx >= 1 else None
        if prev_iter is not None:
            prev_Vl = iteration.Vl_estimate
            prev_tT = iteration.tT
            prev_tW = iteration.tW
            prev_Wl = iteration.Wl_estimate
            prev_Ml = iteration.M

        with self.DBConn(**self.connArgs) as cur:
            cur.execute('''
INSERT INTO tbl_iters(creation_date, totalTime, TOL, bias, stat_error,
Qparams, userdata, iteration_idx, run_id)
VALUES(datetime(), ?, ?, ?, ?, ?, ?, ?, ?)''',
                        _nan2none([iteration.totalTime, iteration.TOL,
                                   iteration.bias, iteration.stat_error])
                        +[_pickle(iteration.Q), _pickle(userdata),
                          iteration_idx, run_id])
            iter_id = cur.getLastRowID()

            # Only add levels that are different from the
            #       previous iteration
            for k in range(0, iteration.lvls_count):
                lvl_data = _nan2none([El[k], Vl[k], Wl[k], tT[k], tW[k], Ml[k]])
                if prev_iter is not None:
                    if k < prev_iter.lvls_count:
                        if np.all(prev_iter.psums_delta[k, :] == iteration.psums_delta[k, :]) and \
                           np.all(prev_iter.psums_fine[k, :] == iteration.psums_fine[k, :]) and \
                           np.all(np.array(lvl_data[1:]) ==
                                  _nan2none([prev_Vl[k],
                                             prev_Wl[k], prev_tT[k], prev_tW[k], prev_Ml[k]])):
                            continue         # Index is repeated as is in this iteration

                lvl = ",".join(["%d|%d" % (i, j) for i, j in
                                enumerate(iteration.lvls_get(k)) if j > base])
                cur.execute('''
INSERT INTO tbl_lvls(lvl, lvl_hash, psums_delta, psums_fine, iter_id,  El, Vl, Wl, tT, tW, Ml)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            [lvl, _md5(lvl),
                             _pickle(iteration.psums_delta[k, :]),
                             _pickle(iteration.psums_fine[k, :]), iter_id]+
                            lvl_data)

    def readRunsByID(self, run_ids):
        from . import mimc
        import re
        lstruns = []
        run_ids = np.array(run_ids).astype(np.int).reshape(-1).tolist()
        if len(run_ids) == 0:
            return lstruns

        with self.DBConn(**self.connArgs) as cur:
            runAll = cur.execute(
                        '''SELECT r.run_id, r.params, r.TOL, r.comment, r.fn, r.tag, r.totalTime
                        FROM tbl_runs r WHERE r.run_id in ?''', [run_ids]).fetchall()
            iterAll = cur.execute('''
SELECT dr.run_id, dr.iter_id, dr.TOL, dr.creation_date,
        dr.totalTime, dr.bias, dr.stat_error, dr.Qparams, dr.userdata,
        dr.iteration_idx FROM tbl_iters dr WHERE dr.run_id in ?
ORDER BY dr.run_id, dr.iteration_idx
''', [run_ids]).fetchall()

            lvlsAll = cur.execute('''
            SELECT dr.iter_id, l.lvl, l.psums_delta, l.psums_fine, l.Ml,
                     l.tT, l.tW, l.Wl, l.Vl
            FROM
            tbl_lvls l INNER JOIN tbl_iters dr ON
            dr.iter_id=l.iter_id INNER JOIN tbl_runs r on r.run_id=dr.run_id
            WHERE dr.run_id in ? ORDER BY dr.iter_id''',
                                  [run_ids]).fetchall()

        dictRuns = dict()
        import dill
        dictLvls = dict()
        dictIters = dict()
        import itertools
        for iter_id, itr in itertools.groupby(lvlsAll, key=lambda x:x[0]):
            dictLvls[iter_id] = list(itr)
        for run_id, itr in itertools.groupby(iterAll, key=lambda x: x[0]):
            dictIters[run_id] = list(itr)

        for run_data in runAll:
            run = mimc.MIMCRun(**_unpickle(run_data[1]).getDict())
            run.db_data = mimc.Bunch()
            run.db_data.finalTOL = run_data[2]
            run.db_data.comment = run_data[3]
            run.db_data.tag = run_data[5]
            run.db_data.totalTime = run_data[6]
            run.db_data.run_id = run_data[0]

            run.setFunctions(**_unpickle(run_data[4], load=dill.load))
            lstruns.append(run)
            if run.db_data.run_id not in dictIters:
                continue
            for i, data in enumerate(dictIters[run.db_data.run_id]):
                iter_id = data[1]
                assert(i == data[9])  # Should be the same as the iteration index
                if run.last_itr is not None:
                    iteration = run.last_itr.next_itr()
                else:
                    iteration = mimc.MIMCItrData(min_dim=run.params.min_dim,
                                                 moments=run.params.moments)
                iteration.TOL = data[2]
                iteration.db_data = mimc.Bunch()
                iteration.db_data.iter_id = iter_id
                iteration.db_data.user_data = _unpickle(data[8])
                iteration.db_data.creation_date = data[3]
                iteration.totalTime = data[4]
                iteration.bias = _none2nan(data[5])
                iteration.stat_error = _none2nan(data[6])
                iteration.Q = _unpickle(data[7])
                run.iters.append(iteration)
                if iter_id not in dictLvls:
                    continue
                for l in dictLvls[iter_id]:
                    t = np.array(map(int, [p for p in re.split(",|\|", l[1]) if p]),
                                 dtype=setutil.ind_t)
                    k = iteration.lvls_find(ind=t[1::2], j=t[::2])
                    if k is None:
                        iteration.lvls_add_from_list(inds=[t[1::2]], j=[t[::2]])
                        k = iteration.lvls_count-1
                    iteration.zero_samples(k)
                    iteration.addSamples(k, M=_none2nan(l[4]),
                                         tT=_none2nan(l[5]),
                                         tW=_none2nan(l[6]),
                                         psums_delta=_unpickle(l[2]),
                                         psums_fine=_unpickle(l[3]))
                    iteration.Wl_estimate[k] = _none2nan(l[7])
                    iteration.Vl_estimate[k] = _none2nan(l[8])
        return lstruns

    def _fetchArray(self, query, params=None):
        with self.DBConn(**self.connArgs) as cur:
            return np.array(cur.execute(query, params if params else []).fetchall())

    def getRunsIDs(self, minTOL=None, maxTOL=None, tag=None,
                   TOL=None, from_date=None, to_date=None,
                   done_flag=None):
        qs = []
        params = []
        if done_flag is not None:
            qs.append('done_flag in ?')
            params.append(np.array(done_flag).astype(np.int).reshape(-1).tolist())
        if tag is not None:
            qs.append('tag LIKE ? ')
            params.append(tag)
        if minTOL is not None:
            qs.append('TOL >= ?')
            params.append(minTOL)
        if maxTOL is not None:
            qs.append('TOL <= ?')
            params.append(maxTOL)
        if TOL is not None:
            qs.append('TOL in ?')
            params.append(np.array(TOL).reshape(-1).tolist())
        if from_date is not None:
            qs.append('creation_date >= ?')
            params.append(from_date)
        if to_date is not None:
            qs.append('creation_date <= ?')
            params.append(to_date)
        wherestr = ("WHERE " + " AND ".join(qs)) if len(qs) > 0 else ''
        query = '''SELECT run_id FROM tbl_runs {wherestr} ORDER BY tag,
        TOL'''.format(wherestr=wherestr)

        ids = self._fetchArray(query, params)
        if ids.size > 0:
            return ids[:, 0]
        return ids

    def readRuns(self, minTOL=None, maxTOL=None, tag=None,
                 TOL=None, from_date=None, to_date=None,
                 done_flag=None):
        runs_ids = self.getRunsIDs(minTOL=minTOL, maxTOL=maxTOL,
                                   tag=tag, TOL=TOL,
                                   from_date=from_date, to_date=to_date,
                                   done_flag=done_flag)
        if len(runs_ids) == 0:
            return []
        return self.readRunsByID(runs_ids)

    def deleteRuns(self, run_ids):
        if len(run_ids) == 0:
            return 0
        with self.DBConn(**self.connArgs) as cur:
            cur.execute("DELETE from tbl_runs where run_id in ?",
                        [np.array(run_ids).astype(np.int).reshape(-1).tolist()])
            return cur.getRowCount()
