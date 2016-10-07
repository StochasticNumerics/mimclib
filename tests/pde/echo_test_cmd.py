#!/usr/bin/python
import numpy as np
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.register('type', 'bool',
                lambda v: v.lower() in ("yes", "true", "t", "1"))
parser.add_argument("-tries", type=int, action="store", default=0)
parser.add_argument("-mimc_dim", type=int, action="store",
                    default=1, help="MIMC dim")

args, unknowns = parser.parse_known_args()

dim = args.mimc_dim
base = "mimc_run.py -mimc_TOL {TOL} -mimc_max_TOL 0.5  -mimc_min_dim {dim} -qoi_seed {seed} -qoi_dim 3 \
-qoi_x0 0.3,0.2,0.6 -qoi_sigma 0.16 -qoi_scale 1 -ksp_rtol 1e-25 \
-mimc_theta 0.2 -qoi_scale 50 -mimc_M0 5 \
-ksp_type gmres -mimc_w {w} -mimc_s {s} -mimc_gamma {gamma} -mimc_beta {beta} \
-mimc_bayes_fit_lvls 3 -mimc_moments 4 \
-mimc_bayesian {bayesian} ".format(bayesian="{bayesian}", TOL="{TOL}",
                                   dim=dim, seed="{seed}",
                                   gamma=" ".join([str(int(3/dim))]*dim),
                                   w=" ".join(["2"]*dim),
                                   s=" ".join(["4"]*dim),
                                   beta=" ".join(["2"]*dim))

base += " ".join(unknowns)

if args.tries == 0:
    cmd_single = "python " + base + " -mimc_verbose 10 -db False "
    print(cmd_single.format(seed=0, bayesian=False, TOL=0.001, dim=args.mimc_dim))
else:
    cmd_multi = "python " + base + " -mimc_verbose 0 -db True -db_tag {tag} "
    TOL = 0.1*np.sqrt(2.)**-16
    for i in range(0, args.tries):
        print cmd_multi.format(bayesian=False, dim=args.mimc_dim,
                               tag="PDE_testcase_dim{}".format(dim), TOL=TOL,
                               seed=np.random.randint(2**32-1))
