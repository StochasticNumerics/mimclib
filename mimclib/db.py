from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cPickle

__all__ = []

def public(sym):
    __all__.append(sym.__name__)
    return sym


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

def _nan2none(x):
    return None if np.isnan(x) else x

class DBConn(object):
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


@public
class MIMCDatabase(object):
    def __init__(self, db='mimc', runTable='tbl_runs', dataTable='tbl_data',
                 lvlTable='tbl_lvls', **kwargs):
        self.DBName = db
        self.runTable = runTable
        self.dataTable = dataTable
        self.lvlTable = lvlTable
        kwargs["db"] = db
        self.connArgs = kwargs.copy()

    def DBCreationScript(self, drop_db=False):
        script = ""
        if drop_db:
            script += "DROP DATABASE IF EXISTS {DBName};".format(DBName=self.DBName)
        script += '''
CREATE DATABASE IF NOT EXISTS {DBName};
USE {DBName};
CREATE TABLE IF NOT EXISTS {runTable} (
    run_id                INTEGER PRIMARY KEY AUTO_INCREMENT NOT NULL,
    creation_date           DATETIME NOT NULL,
    TOL                   REAL NOT NULL,
    done_flag            INTEGER NOT NULL,
    totalTime               REAL,
    tag                   VARCHAR(128) NOT NULL,
    params                mediumblob,
    fnNorm                    mediumblob,
    comment               TEXT
);
CREATE VIEW vw_runs AS SELECT run_id, creation_date, TOL, done_flag, tag, totalTime, comment FROM {runTable};

CREATE TABLE IF NOT EXISTS {dataTable} (
    data_id                 INTEGER PRIMARY KEY AUTO_INCREMENT NOT NULL,
    run_id                  INTEGER NOT NULL,
    TOL                     REAL,
    bias                    REAL,
    stat_error              REAL,
    creation_date           DATETIME NOT NULL,
    totalTime               REAL,
    Qparams                 mediumblob,
    userdata                mediumblob,
    iteration_idx           INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES {runTable}(run_id) ON DELETE CASCADE,
    UNIQUE KEY idx_itr_idx (run_id, iteration_idx)
);
CREATE VIEW vw_data AS SELECT data_id, run_id, TOL,
creation_date, bias, stat_error, totalTime, iteration_idx FROM {dataTable};

CREATE TABLE IF NOT EXISTS {lvlTable} (
    data_id       INTEGER NOT NULL,
    lvl           text NOT NULL,
    lvl_hash      varchar(35) NOT NULL,
    El            REAL,
    Vl            REAL,
    Wl            REAL,
    Tl            REAL,
    Ml            INTEGER,
    psums_delta   mediumblob,
    psums_fine    mediumblob,
    FOREIGN KEY (data_id) REFERENCES {dataTable}(data_id) ON DELETE CASCADE,
    UNIQUE KEY idx_run_lvl (data_id, lvl_hash)
);

CREATE VIEW vw_lvls AS SELECT data_id, lvl,
El, Vl, Wl, Tl, Ml FROM {lvlTable};

-- CREATE USER 'USER'@'%';
-- GRANT ALL PRIVILEGES ON *.* TO 'USER'@'%' WITH GRANT OPTION;
'''.format(DBName=self.DBName, runTable=self.runTable,
           dataTable=self.dataTable, lvlTable=self.lvlTable)

        return script

    def createRun(self, tag, TOL=None, params=None, fnNorm=None,
                  mimc_run=None, comment=""):
        TOL = TOL or mimc_run.params.TOL
        params = params or mimc_run.params
        fnNorm = fnNorm or mimc_run.fn.Norm
        import dill
        with DBConn(**self.connArgs) as cur:
            cur.execute('''
            INSERT INTO {runTable}(creation_date, TOL, tag, params, fnNorm, done_flag, comment)
            VALUES(datetime(), ?, ?, ?, ?, -1, ?)'''.format(runTable=self.runTable),
                        [TOL, tag, _pickle(params), _pickle(fnNorm, dump=dill.dump), comment])
            return cur.getLastRowID()

    def markRunDone(self, run_id, flag, totalTime=None, comment=''):
        with DBConn(**self.connArgs) as cur:
            cur.execute(''' UPDATE {runTable} SET done_flag=?, totalTime=?,
            comment = CONCAT(comment,  ?)
            WHERE run_id=?'''.format(runTable=self.runTable),
                        [flag, totalTime, comment, run_id])

    def markRunSuccessful(self, run_id, totalTime=None, comment=''):
        self.markRunDone(run_id, flag=1, comment=comment, totalTime=totalTime)

    def markRunFailed(self, run_id, totalTime=None, comment='', add_exception=True):
        if add_exception:
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            if exc_obj is not None:
                comment += "{}: {}".format(exc_type.__name__, exc_obj)
        self.markRunDone(run_id, flag=0, comment=comment, totalTime=totalTime)

    def writeRunData(self, run_id, mimc_run, iteration_idx, TOL,
                     totalTime, userdata=None):
        base = 0
        El = mimc_run.fn.Norm(mimc_run.data.calcDeltaEl())
        Vl = mimc_run.Vl_estimate
        Tl = mimc_run.data.calcTl()
        Wl = mimc_run.Wl_estimate
        Ml = mimc_run.data.M

        with DBConn(**self.connArgs) as cur:
            cur.execute('''
INSERT INTO {dataTable}(creation_date, totalTime, TOL, bias, stat_error,
Qparams, userdata, iteration_idx, run_id)
VALUES(datetime(), ?, ?, ?, ?, ?, ?, ?, ?)'''.format(dataTable=self.dataTable),
                        [_nan2none(totalTime),
                         _nan2none(TOL),
                         _nan2none(mimc_run.bias),
                         _nan2none(mimc_run.stat_error),
                         _pickle(mimc_run.Q), _pickle(userdata),
                         iteration_idx, run_id])
            data_id = cur.getLastRowID()
            for k in range(0, len(mimc_run.data.lvls)):
                lvl = ",".join(["%d|%d" % (i, j) for i, j in
                                enumerate(mimc_run.data.lvls[k]) if j > base])
                cur.execute('''
INSERT INTO {lvlTable}(lvl, lvl_hash, El, Vl, Wl, Tl, Ml, psums_delta, psums_fine, data_id)
VALUES(?, md5(?), ?, ?, ?, ?, ?, ?, ?, ?)
'''.format(lvlTable=self.lvlTable, lvl=lvl),
                            [lvl, lvl, _nan2none(El[k]), _nan2none(Vl[k]), _nan2none(Wl[k]), _nan2none(Tl[k]), _nan2none(Ml[k]),
                             _pickle(mimc_run.data.psums_delta[k, :]),
                             _pickle(mimc_run.data.psums_fine[k, :]), data_id])

    def readRunsByID(self, run_ids):
        from . import mimc
        import re
        lstvalues = []
        run_ids = np.array(run_ids).astype(np.int).reshape(-1).tolist()
        if len(run_ids) == 0:
            return lstvalues

        with DBConn(**self.connArgs) as cur:
            runAll = cur.execute(
                        '''SELECT r.run_id, r.params, r.TOL, r.comment, count(*), r.fnNorm, r.tag
                        FROM {runTable} r INNER JOIN {dataTable} dr ON
                        r.run_id = dr.run_id WHERE r.run_id in ? GROUP BY
                        r.run_id, r.params, r.TOL, r.comment, r.fnNorm'''.
                        format(runTable=self.runTable,
                               dataTable=self.dataTable), [run_ids]).fetchall()

            dataAll = cur.execute('''
SELECT dr.data_id, dr.run_id, dr.TOL, dr.creation_date,
        dr.totalTime, dr.bias, dr.stat_error, dr.Qparams, dr.userdata,
        dr.iteration_idx FROM {dataTable} dr WHERE dr.run_id in ?
'''.format(dataTable=self.dataTable), [run_ids]).fetchall()

            lvlsAll = cur.execute('''
            SELECT dr.data_id, l.lvl, l.psums_delta, l.psums_fine, l.Ml,
                     l.Tl, l.Wl, l.Vl
            FROM
            {lvlTable} l INNER JOIN {dataTable} dr ON
            dr.data_id=l.data_id INNER JOIN {runTable} r on r.run_id=dr.run_id
            WHERE dr.run_id in ? ORDER BY dr.data_id'''.
                                  format(lvlTable=self.lvlTable,
                                         dataTable=self.dataTable,
                                         runTable=self.runTable),
                                  [run_ids]).fetchall()

        dictRuns = dict()
        import dill
        for run in runAll:
            dictRuns[run[0]] = [_unpickle(run[1]), run[2], run[3],
                                run[4], _unpickle(run[5], load=dill.load),
                                run[6]]

        dictLvls = dict()
        import itertools
        for run_id, itr in itertools.groupby(lvlsAll, key=lambda x:x[0]):
            psums_delta, psums_fine, lvls, Ml, Tl, Wl, Vl = [], [], [], [], [], [], []
            for r in itr:
                t = np.array(map(int, [p for p in re.split(",|\|", r[1]) if p]),
                             dtype=np.uint32)
                ind = np.zeros(r[-1], dtype=np.uint)
                ind[t[::2]] = t[1::2]
                lvls.append(ind.tolist())
                psums_delta.append(_unpickle(r[2]))
                psums_fine.append(_unpickle(r[3]))
                Ml.append(r[4])
                Tl.append(r[5])
                Wl.append(r[6])
                Vl.append(r[7])
            dictLvls[run_id] = [psums_delta, psums_fine, lvls, Ml, Tl, Wl, Vl]

        for data in dataAll:
            val = dict()
            run_id = data[1]
            data_id = data[0]
            run_params = dictRuns[run_id]
            val["data_id"] = data_id
            val["finalTOL"] = run_params[1]
            val["comment"] = run_params[2]
            val["total_iterations"] = run_params[3]
            val["tag"] = run_params[5]
            val["TOL"] = data[2]
            val["creation_date"] = data[3]
            val["totalTime"] = data[4]
            val["user_data"] = _unpickle(data[8])
            val["iteration_index"] = data[9]

            psums_delta, psums_fine, lvls, Ml, Tl, Wl, Vl = dictLvls[data_id]

            lvls = np.array(lvls)
            sort_rows = lambda a: np.argsort(a.view([('',a.dtype)]*a.shape[1]),0).T[0]
            ind = sort_rows(lvls)

            old_data = mimc.MIMCData(min_dim=run_params[0].dim,
                                     lvls=lvls[ind],
                                     psums_delta=np.array(psums_delta)[ind],
                                     psums_fine=np.array(psums_fine)[ind],
                                     M=np.array(Ml)[ind],
                                     t=np.array(Ml)[ind] * np.array(Tl)[ind])
            run = mimc.MIMCRun(old_data=old_data,
                               **run_params[0].getDict())
            run.setFunctions(Norm=run_params[4].getDict())
            run.bias = data[5]
            run.stat_error = data[6]
            run.Q = _unpickle(data[7])
            val["run"] = run
            run.Vl_estimate = np.array(Vl)[ind]
            run.Wl_estimate = np.array(Wl)[ind]
            lstvalues.append(mimc.MyDefaultDict(**val))
        return lstvalues

    def _fetchArray(self, query, params=None):
        with DBConn(**self.connArgs) as cur:
            dataAll = cur.execute(query, params if params else [])
            return np.array(dataAll.fetchall())

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
        query = '''SELECT run_id FROM {runTable} {wherestr} ORDER BY tag,
        TOL'''.format(runTable=self.runTable, wherestr=wherestr)

        ids = self._fetchArray(query, params)
        if ids.size > 0:
            return ids[:, 0]
        return ids

    def getRunDataIDs(self, run_ids):
        run_ids = np.array(run_ids).astype(np.int).reshape(-1).tolist()
        if len(run_ids) == 0:
            raise ValueError("Must have at least one run")

        query = '''SELECT DISTINCT data_id FROM {dataTable} rd WHERE rd.run_id in ? '''.format(dataTable=self.dataTable)
        ids = self._fetchArray(query, params=[run_ids])
        if ids.size > 0:
            return ids[:, 0]
        return ids

    def filterRuns(self, runs, iteration_idx=None):
        if iteration_idx is not None:
            ii = np.array(iteration_idx).reshape((-1,))
            from_end = ii<0
            runs = [d for d in runs if d.iteration_index in
                    d.total_iterations*from_end + ii]
        return runs

    def readRuns(self, minTOL=None, maxTOL=None, tag=None,
                 TOL=None, from_date=None, to_date=None,
                 done_flag=None, iteration_idx=None):
        runs_ids = self.getRunsIDs(minTOL=minTOL, maxTOL=maxTOL,
                                   tag=tag, TOL=TOL,
                                   from_date=from_date, to_date=to_date,
                                   done_flag=done_flag)
        if len(runs_ids) == 0:
            return []
        return self.filterRuns(self.readRunsByID(runs_ids),
                               iteration_idx=iteration_idx)

    def deleteRuns(self, run_ids):
        if len(run_ids) == 0:
            return 0
        with DBConn(**self.connArgs) as cur:
            cur.execute("DELETE from {runTable} where run_id in ?".format(runTable=self.runTable),
                        [np.array(run_ids).astype(np.int).reshape(-1).tolist()])
            return cur.getRowCount()
