#!/usr/bin/python
import numpy as np
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.register('type', 'bool',
                lambda v: v.lower() in ("yes", "true", "t", "1"))
parser.add_argument("-multi", type="bool", action="store",
                    default=False, help="True output a single command")
parser.add_argument("-db_host", type=str, action="store",
                    default="localhost",
                    help="True output a single command")

args = parser.parse_known_args()[0]

base = "run.py -mimc_TOL {TOL} -mimc_max_TOL 0.5  \
-qoi_sigma 0.1 -qoi_mu 1 -qoi_seed {seed} \
-mimc_dim 1 -mimc_w 1 -mimc_s 1 -mimc_gamma 1 -mimc_beta 2 \
-mimc_theta {theta} -mimc_bayesian {bayesian} "

cmd_multi = "python " + base + "-mimc_verbose False -db True -db_tag {tag} " + " -db_host {} ".format(args.db_host)
cmd_single = "python " + base + " -mimc_verbose True -db False "

if not args.multi:
    print(cmd_single.format(seed=0, bayesian=True, TOL=0.001, theta="0.2"))
else:
    realizations = 35
    TOLs = 0.1*np.sqrt(2.)**-np.arange(0., 10.)
    for TOL in TOLs:
        for i in range(0, realizations):
            print cmd_multi.format(bayesian=False, tag="GBM_test", TOL=TOL,
                                   seed=np.random.randint(2**32-1), theta="0.2")

