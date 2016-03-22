from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


__all__ = []

def public(sym):
    __all__.append(sym.__name__)
    return sym


def _pickle(obj):
    import io
    import cPickle
    import MySQLdb
    with io.BytesIO() as f:
        cPickle.dump(obj, f, protocol=2)
        f.seek(0)
        return MySQLdb.Binary(f.read())


def _unpickle(obj):
    import io
    import cPickle
    with io.BytesIO(obj) as f:
        return cPickle.load(f)


class DBConn(object):
    def __init__(self, **kwargs):
        self.connArgs = kwargs

    def __enter__(self):
        import MySQLdb
        self.conn = MySQLdb.connect(**self.connArgs)
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
    TOL                   REAL,
    done_flag            INTEGER NOT NULL,
    dim                   INTEGER,
    tag                   VARCHAR(128),
    params                mediumblob
);
CREATE VIEW vw_runs AS SELECT run_id, creation_date, TOL, done_flag, dim, tag FROM {runTable};

CREATE TABLE IF NOT EXISTS {dataTable} (
    data_id                 INTEGER PRIMARY KEY AUTO_INCREMENT NOT NULL,
    run_id                  INTEGER NOT NULL,
    TOL                     REAL NOT NULL,
    bias                    REAL,
    stat_error              REAL,
    creation_date           DATETIME NOT NULL,
    totalTime               REAL NOT NULL,
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
    psums         mediumblob,
    FOREIGN KEY (data_id) REFERENCES {dataTable}(data_id) ON DELETE CASCADE,
    UNIQUE KEY idx_run_lvl (data_id, lvl_hash)
);

CREATE VIEW vw_lvls AS SELECT data_id, lvl,
El, Vl, Wl, Tl, Ml FROM {lvlTable};
'''.format(DBName=self.DBName, runTable=self.runTable,
           dataTable=self.dataTable, lvlTable=self.lvlTable)

        return script

    def createRun(self, tag, TOL=None, dim=None, params=None,
                  mimc_run=None):
        TOL = TOL or mimc_run.params.TOL
        params = params or mimc_run.params
        dim = dim or mimc_run.data.dim
        with DBConn(**self.connArgs) as cur:
            cur.execute('''
INSERT INTO {runTable}(creation_date, TOL, tag, dim, params, done_flag)
VALUES(datetime(), ?, ?, ?, ?, -1)'''.format(runTable=self.runTable),
                        [TOL, tag, dim, _pickle(params)])
            return cur.getLastRowID()

    def markRunDone(self, run_id, flag=1):
        with DBConn(**self.connArgs) as cur:
            cur.execute(''' UPDATE {runTable} SET done_flag={flag}
            WHERE run_id={run_id}'''.format(runTable=self.runTable,
                                            flag=flag, run_id=run_id))

    def writeRunData(self, run_id, mimc_run, iteration_idx, TOL,
                     totalTime, userdata=None):
        base = 0
        El = mimc_run.data.calcEl()
        Vl = mimc_run.Vl_estimate
        Tl = mimc_run.data.calcTl()
        Wl = mimc_run.Wl_estimate
        Ml = mimc_run.data.M

        with DBConn(**self.connArgs) as cur:
            cur.execute('''
INSERT INTO {dataTable}(creation_date, totalTime, TOL, bias, stat_error,
Qparams, userdata, iteration_idx, run_id)
VALUES(datetime(), ?, ?, ?, ?, ?, ?, ?, ?)'''.format(dataTable=self.dataTable),
                        [totalTime, TOL, mimc_run.bias, mimc_run.stat_error,
                         _pickle(mimc_run.Q), _pickle(userdata),
                         iteration_idx, run_id])
            data_id = cur.getLastRowID()
            for k in range(0, len(mimc_run.data.lvls)):
                lvl = ",".join(["%d|%d" % (i, j) for i, j in
                                enumerate(mimc_run.data.lvls[k]) if j > base])
                cur.execute('''
INSERT INTO {lvlTable}(lvl, lvl_hash, El, Vl, Wl, Tl, Ml, psums, data_id)
VALUES(?, md5('{lvl}'), ?, ?, ?, ?, ?, ?, ?)
'''.format(lvlTable=self.lvlTable, lvl=lvl),
                            [lvl, El[k], Vl[k], Wl[k], Tl[k], Ml[k],
                             _pickle(mimc_run.data.psums[k, :]), data_id])

    def readRunData(self, data_ids):
        from . import mimc
        import re
        lstvalues = []
        dictParams = dict()
        with DBConn(**self.connArgs) as cur:
            for data_id in data_ids:
                val = dict()
                dataAll = cur.execute('''
SELECT dr.run_id, dr.TOL, dr.creation_date, dr.totalTime, dr.bias,
dr.stat_error, dr.Qparams, dr.userdata, dr.iteration_idx
FROM {dataTable} dr WHERE data_id=?'''.format(dataTable=self.dataTable), [data_id]).fetchall()
                run_id = dataAll[0][0]
                if run_id not in dictParams:
                    dataTmp = dictParams[run_id] = cur.execute(
                        '''SELECT params, TOL FROM {runTable} WHERE run_id=?'''.
                        format(runTable=self.runTable), [run_id]).fetchall()
                    dataTmp2 = cur.execute(
                        '''SELECT count(*) FROM {dataTable} WHERE run_id=?'''.
                        format(dataTable=self.dataTable), [run_id]).fetchall()
                    dictParams[run_id] = [_unpickle(dataTmp[0][0]),
                                          dataTmp[0][1], dataTmp2[0][0]]
                val["data_id"] = data_id
                val["finalTOL"] = dictParams[run_id][1]
                val["total_iterations"] = dictParams[run_id][2]
                val["TOL"] = dataAll[0][1]
                val["creation_date"] = dataAll[0][2]
                val["totalTime"] = dataAll[0][3]
                run = mimc.MIMCRun(**dictParams[run_id][0].getDict())
                run.bias = dataAll[0][4]
                run.stat_error = dataAll[0][5]
                run.Q = _unpickle(dataAll[0][6])
                val["run"] = run
                val["userData"] = _unpickle(dataAll[0][7])
                val["iteration_index"] = dataAll[0][8]

                dataAll = cur.execute('''
SELECT lvl, psums, Ml, Tl, Wl, Vl
FROM {lvlTable} WHERE data_id=?'''.format(lvlTable=self.lvlTable), [data_id]).fetchall()
                psums, lvls, Ml, Tl, Wl, Vl = [], [], [], [], [], []
                for r in dataAll:
                    t = np.array(map(int, [p for p in re.split(",|\|", r[0]) if p]),
                                 dtype=np.uint32)
                    ind = np.zeros(run.params.dim, dtype=np.uint)
                    ind[t[::2]] = t[1::2]
                    lvls.append(ind.tolist())
                    psums.append(_unpickle(r[1]))
                    Ml.append(r[2])
                    Tl.append(r[3])
                    Wl.append(r[4])
                    Vl.append(r[5])

                lvls = np.array(lvls)
                sort_rows = lambda a: np.argsort(a.view([('',a.dtype)]*a.shape[1]),0).T[0]
                ind = sort_rows(lvls)
                run.all_data.lvls = run.data.lvls = lvls[ind]
                run.all_data.psums = run.data.psums = np.array(psums)[ind]
                run.all_data.M = run.data.M = np.array(Ml)[ind]
                run.all_data.t = run.data.t = run.data.M * np.array(Tl)[ind]
                run.Vl_estimate = np.array(Vl)[ind]
                run.Wl_estimate = np.array(Wl)[ind]
                lstvalues.append(mimc.MyDefaultDict(**val))
        return lstvalues

    def _fetchArray(self, query):
        with DBConn(**self.connArgs) as cur:
            dataAll = cur.execute(query)
            return np.array(dataAll.fetchall())

    def getRunsIDs(self, minTOL=None, maxTOL=None, dim=None, tag=None):
        qs = []
        if dim is not None:
            qs.append('dim in ({})'.
                      format(','.join(map(str, np.array(dim).
                                          astype(np.int).reshape(-1)))))
        if tag is not None:
            qs.append('''tag LIKE '{}' '''.format(tag))
        if minTOL is not None:
            qs.append('TOL>=({})'.format(minTOL))
        if maxTOL is not None:
            qs.append('TOL<=({})'.format(maxTOL))

        wherestr = ("WHERE " + " AND ".join(qs)) if len(qs) > 0 else ''
        ids = self._fetchArray("SELECT DISTINCT run_id FROM {runTable}\
{wherestr} ORDER BY tag, dim, TOL".format(runTable=self.runTable,
                                          wherestr=wherestr))
        if ids.size > 0:
            return ids[:, 0]
        return ids

    def getRunDataIDs(self, run_id=None, minTOL=None, maxTOL=None,
                      dim=None, tag=None, done_flag=None):
        if (run_id is not None) and (dim is not None or tag is not None):
            raise Exception("Cannot specify dimensions and \
tag after specifying run_id")

        qs = []
        if dim is not None:
            qs.append('r.dim in ({})'.
                      format(','.join(map(str, np.array(dim).
                                          astype(np.int).reshape(-1)))))
        if tag is not None:
            qs.append('''r.tag LIKE '{}' '''.format(tag))

        if run_id is not None:
            qs.append('r.run_id in ({})'.
                      format(','.join(map(str, np.array(run_id).
                                          astype(np.int).reshape(-1)))))

        if done_flag is not None:
            qs.append('r.done_flag in ({})'.
                      format(','.join(map(str, np.array(done_flag).
                                          astype(np.int).reshape(-1)))))

        if minTOL is not None:
            qs.append('rd.TOL>=({})'.format(minTOL))
        if maxTOL is not None:
            qs.append('rd.TOL<=({})'.format(maxTOL))

        wherestr = ("WHERE " + " AND ".join(qs)) if len(qs) > 0 else ''
        joinstr = '' if dim is None and tag is None else\
                  ("INNER JOIN {runTable} r ON r.run_id = rd.run_id".
                   format(runTable=self.runTable))

        ids = self._fetchArray("SELECT DISTINCT data_id FROM {dataTable} rd \
{joinstr} {wherestr}".format(dataTable=self.dataTable,
                             joinstr=joinstr,
                             wherestr=wherestr))
        if ids.size > 0:
            return ids[:, 0]
        return ids
