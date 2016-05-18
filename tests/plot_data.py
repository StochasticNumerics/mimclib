#!/usr/bin/python
import mimclib.db as mimcdb
import mimclib.plot as miplot
import warnings
import os
warnings.formatwarning = lambda msg, cat, filename, lineno, line: \
                             "{}:{}: ({}) {}\n".format(os.path.basename(filename),
                                                       lineno, cat.__name__, msg)
try:
    from matplotlib.cbook import MatplotlibDeprecationWarning
    warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
except:
    pass   # Ignore


def addExtraArguments(parser):
    parser.register('type', 'bool', lambda v: v.lower() in ("yes",
                                                            "true",
                                                            "t", "1"))
    parser.add_argument("-db_name", type=str, action="store",
                        help="Database Name")
    parser.add_argument("-db_user", type=str, action="store",
                        help="Database User")
    parser.add_argument("-db_host", type=str, action="store",
                        help="Database Host")
    parser.add_argument("-db_tag", type=str, action="store",
                        help="Database Tag")
    parser.add_argument("-qoi_exact", type=float, action="store",
                        help="Exact value")
    parser.add_argument("-o", type=str, default="mimc_results.pdf",
                        action="store", help="Output file")
    parser.add_argument("-cmd", type=str, action="store",
                        help="Command to execute after plotting")


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=True)
    addExtraArguments(parser)
    import mimclib.test
    args = mimclib.test.parse_known_args(parser)
    db_args = dict()
    if args.db_name is not None:
        db_args["db"] = args.db_name
    print(args.db_name)
    if args.db_user is not None:
        db_args["user"] = args.db_user
    if args.db_host is not None:
        db_args["host"] = args.db_host
    db = mimcdb.MIMCDatabase(**db_args)

    if args.db_tag is None:
        warnings.warn("You did not select a database tag!!")
    run_data = db.readRuns(db.getRunsIDs(tag=args.db_tag, done_flag=1))
    run_data = [d for d in run_data if d.iteration_index+1 ==
                d.total_iterations]
    if len(run_data) == 0:
        raise Exception("No runs!!!")
    miplot.genPDFBooklet(run_data, fileName=args.o, exact=args.qoi_exact)

    if args.cmd is not None:
        import os
        os.system(args.cmd.format(args.o))

if __name__ == "__main__":
    main()
