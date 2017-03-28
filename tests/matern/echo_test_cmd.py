#!/usr/bin/python
import numpy as np
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.register('type', 'bool',
                lambda v: v.lower() in ("yes", "true", "t", "1"))
parser.add_argument("-db", type="bool", action="store", default=False)
parser.add_argument("-qoi_dim", type=int, action="store",
                    default=1, help="MIMC dim")
parser.add_argument("-qoi_df_nu", type=float, action="store",
                    default=3.5, help="MIMC dim")

args, unknowns = parser.parse_known_args()

if args.qoi_dim:
    base = "\
 misc_run.py -mimc_TOL {TOL} -qoi_seed 0 \
-qoi_problem 0 -qoi_sigma 0.2 \
-mimc_min_dim {qoi_dim} -qoi_dim {qoi_dim} -qoi_df_nu {qoi_df_nu} \
-qoi_x0 0.3 0.4 0.6 -ksp_rtol 1e-25 -ksp_type gmres  \
-qoi_a0 0 -qoi_f0 1 \
-qoi_scale 10 -qoi_df_sig 0.5 -mimc_M0 1 \
-mimc_beta {beta} -mimc_gamma {gamma} -mimc_h0inv 3 \
-mimc_bayes_fit_lvls 3 -mimc_moments 1 -mimc_bayesian False \
".format(TOL="{TOL}", qoi_df_nu=args.qoi_df_nu, qoi_dim=args.qoi_dim,
         beta=" ".join([str("2")]*args.qoi_dim),
         gamma=" ".join([str("1")]*args.qoi_dim))
else:
    assert False

base += " ".join(unknowns)

if not args.db:
    cmd_single = "python " + base + " -mimc_verbose 10 -db False "
    print(cmd_single.format(TOL=0.001))
else:
    cmd_multi = "python " + base + " -mimc_verbose 0 -db True -db_tag {tag} "
    print cmd_multi.format(tag="misc_matern_d{:d}_nu{:.2g}".format(args.qoi_dim, args.qoi_df_nu), TOL=1e-10)
