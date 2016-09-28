#!/usr/bin/python
import numpy as np
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.register('type', 'bool',
                lambda v: v.lower() in ("yes", "true", "t", "1"))
parser.add_argument("-multi", type="bool", action="store",
                    default=False, help="True output a single command")
parser.add_argument("-mimc_dim", type=int, action="store",
                    default=1, help="MIMC dim")

args, unknowns = parser.parse_known_args()

dim = args.mimc_dim
if dim == 1:
    base = "mimc_run.py -mimc_TOL {TOL} -mimc_max_TOL 0.5  -mimc_dim 1 -qoi_seed {seed} \
-mimc_theta 0.2 -mimc_M0 25 \
-mimc_w 1 -mimc_s 2 -mimc_gamma 3 -mimc_beta 2 -mimc_h0inv 5 \
-mimc_bayes_fit_lvls 3 -mimc_moments 4 \
-mimc_bayesian {bayesian} ".format(bayesian="{bayesian}", TOL="{TOL}",
                                   seed="{seed}")
elif dim == 2:
    base = "mimc_run.py -mimc_TOL {TOL} -mimc_max_TOL 0.5  -mimc_dim 2 -qoi_seed {seed} \
-mimc_theta 0.2 -mimc_M0 25 -mimc_h0inv 5 5 \
-mimc_w 1 1 -mimc_s 2 2 -mimc_gamma 2 1 -mimc_beta 2 2 \
-mimc_bayes_fit_lvls 3 -mimc_moments 4 \
-mimc_bayesian {bayesian} ".format(bayesian="{bayesian}", TOL="{TOL}", seed="{seed}")
else:
    assert False, "Dim must be 1 or 2"
base += " ".join(unknowns)

cmd_multi = "python " + base + " -mimc_verbose False -db True -db_tag {tag} "
cmd_single = "python " + base + " -mimc_verbose True -db False "

if not args.multi:
    print(cmd_single.format(seed=0, bayesian=False, TOL=0.001, dim=args.mimc_dim))
else:
    realizations = 35
    TOLs = 0.05*np.sqrt(2.)**-np.arange(0., 24.)
    for TOL in TOLs:
        for i in range(0, realizations):
            print cmd_multi.format(bayesian=False, dim=args.mimc_dim,
                                   tag="particle_{}".format(dim), TOL=TOL,
                                   seed=np.random.randint(2**32-1))
