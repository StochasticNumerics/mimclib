#!/usr/bin/python
import numpy as np
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.register('type', 'bool',
                lambda v: v.lower() in ("yes", "true", "t", "1", "y"))
parser.add_argument("-multi", type="bool", action="store",
                    default=False, help="True output a single command")

args, unknowns = parser.parse_known_args()

base = "mimc_run.py -mimc_TOL {TOL} -mimc_max_TOL 0.5  \
-qoi_sigma 0.1 -qoi_mu 1 -qoi_seed {seed} -mimc_moments 4 \
-mimc_dim 1 -mimc_w 1 -mimc_s 1 -mimc_gamma 1 -mimc_beta 2 -mimc_confidence 0.99 \
-mimc_theta {theta} -mimc_bayesian {bayesian} -mimc_bayes_fit_lvls 4 \
-mimc_bayes_k1 0.005 "


base += " ".join(unknowns)
cmd_multi = "python " + base + " -mimc_verbose False -db True -db_tag {tag} "
cmd_single = "python " + base + " -mimc_verbose True -db False "

if not args.multi:
    print(cmd_single.format(seed=0, bayesian=True, TOL=0.001, theta="0.2"))
else:
    realizations = 100
    TOLs = 0.1*np.sqrt(2.)**-np.arange(0., 16.)
    for TOL in TOLs:
        for i in range(0, realizations):
            print cmd_multi.format(bayesian=True, tag="GBM_testcase", TOL=TOL,
                                   seed=np.random.randint(2**32-1), theta="0.2")
