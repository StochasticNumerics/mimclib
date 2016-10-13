#!/usr/bin/python
import numpy as np
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.register('type', 'bool',
                lambda v: v.lower() in ("yes", "true", "t", "1"))
parser.add_argument("-db", type="bool", action="store", default=False)
parser.add_argument("-db_tag", type=str, action="store", default=None)
parser.add_argument("-qoi_dim", type=int, action="store", default=10)
parser.add_argument("-qoi_func", type=int, action="store", default=1)

args, unknowns = parser.parse_known_args()

if args.qoi_dim:
    base = "\
 mimc_run.py -mimc_TOL {TOL} -qoi_seed 0 -mimc_min_dim {qoi_dim} -qoi_dim {qoi_dim}  \
-mimc_M0 1 -mimc_moments 1 -mimc_bayesian False -qoi_func {qoi_func} \
".format(TOL="{TOL}",
         qoi_dim=args.qoi_dim,
         qoi_func=args.qoi_func)
else:
    assert False

base += " ".join(unknowns)

if not args.db:
    cmd_single = "python " + base + " -mimc_verbose 10 -db False "
    print(cmd_single.format(TOL=0.001))
else:
    cmd_multi = "python " + base + " -mimc_verbose 0 -db True -db_tag {tag} "
    db_tag = args.db_tag
    if db_tag is None:
        db_tag = "sc_d{:d}_fn{:.2g}".format(args.qoi_dim, args.qoi_func)
    print cmd_multi.format(tag=db_tag, TOL=1e-10)
