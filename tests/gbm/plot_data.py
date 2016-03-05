#!/usr/bin/python
import mimclib.db as mimcdb
import mimclib.plot as miplot

import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
warnings.simplefilter('ignore', MatplotlibDeprecationWarning)

def addExtraArguments(parser):
    def str2bool(v):
        # susendberg's function
        return v.lower() in ("yes", "true", "t", "1")
    parser.register('type', 'bool', str2bool)
    parser.add_argument("-db_user", type=str, default=None,
                        action="store", help="Database User")
    parser.add_argument("-db_host", type=str, default='localhost',
                        action="store", help="Database Host")
    parser.add_argument("-db_tag", type=str, default="NoTag",
                        action="store", help="Database Tag")
    parser.add_argument("-o", type=str, default="out.pdf",
                        action="store", help="Output file")


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=True)
    addExtraArguments(parser)
    args = parser.parse_known_args()[0]
    db = mimcdb.MIMCDatabase(user=args.db_user,
                             host=args.db_host)

    import numpy as np
    run_data = db.readRunData(db.getRunDataIDs(tag=args.db_tag, done_flag=[1]))
    if len(run_data) == 0:
        raise Exception("No runs!!!")
    miplot.genPDFBooklet(args.o, run_data, exact=np.exp(1.),
                         var_ref_rate=np.log(2), expect_ref_rate=np.log(2))

if __name__ == "__main__":
    main()
