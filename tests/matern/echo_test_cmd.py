#!/usr/bin/python
import numpy as np
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.register('type', 'bool',
                lambda v: v.lower() in ("yes", "true", "t", "1"))
parser.add_argument("-tries", type="bool", action="store",
                    default=False, help="True output a single command")
parser.add_argument("-mimc_dim", type=int, action="store",
                    default=1, help="MIMC dim")
parser.add_argument("-qoi_df_nu", type=float, action="store",
                    default=3.5, help="MIMC dim")
parser.add_argument("-qoi_dim", type=int, action="store",
                    default=1, help="MIMC dim")

args, unknowns = parser.parse_known_args()

dim = args.mimc_dim
if dim == 1:
    base = "\
 mimc_run.py -mimc_TOL {TOL} -mimc_max_TOL 0.5 -qoi_seed {seed} \
-qoi_problem 0 -qoi_sigma 0.2 \
-mimc_min_dim 1 -qoi_dim {qoi_dim} -qoi_df_nu {qoi_df_nu} \
-qoi_x0 0.3 0.4 0.6 -ksp_rtol 1e-25 -ksp_type gmres  \
-qoi_a0 0 -qoi_f0 1 \
-qoi_scale 10 -qoi_df_sig 0.5 -mimc_theta 0.2 -mimc_M0 1 \
-mimc_w 1 -mimc_s 2 -mimc_gamma 3 -mimc_beta 2 -mimc_h0inv 3 \
-mimc_bayes_fit_lvls 3 -mimc_moments 1 -mimc_bayesian False \
".format(bayesian="{bayesian}", TOL="{TOL}", seed="{seed}",
         qoi_df_nu=args.qoi_df_nu, qoi_dim=args.qoi_dim)
else:
    assert False

base += " ".join(unknowns)

if args.tries == 0:
    cmd_single = "python " + base + " -mimc_verbose True -db False "
    print(cmd_single.format(seed=0, bayesian=False, TOL=0.001, dim=args.mimc_dim))
else:
    cmd_multi = "python " + base + " -mimc_verbose False -db True -db_tag {tag} "
    TOLs = 0.05*np.sqrt(2.)**-np.arange(0., 24.)
    for TOL in TOLs:
        for i in range(0, args.tries):
            print cmd_multi.format(bayesian=False, dim=args.mimc_dim,
                                   tag="particle_{}".format(dim), TOL=TOL,
                                   seed=np.random.randint(2**32-1))
