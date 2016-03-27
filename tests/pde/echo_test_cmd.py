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
parser.add_argument("-db_host", type=str, action="store",
                    default="localhost",
                    help="True output a single command")

args = parser.parse_known_args()[0]
dim = args.mimc_dim
base = "run.py -mimc_TOL {TOL} -mimc_max_TOL 0.5  -mimc_dim {dim} -qoi_seed {seed} -qoi_dim 3 \
-qoi_x0 0.3,0.2,0.6 -qoi_sigma 0.16 -qoi_scale 1 -ksp_rtol 1e-25 \
-mimc_theta 0.2 -qoi_scale 50 \
-ksp_type gmres -mimc_w {w} -mimc_s {s} -mimc_gamma {gamma} -mimc_beta {beta} \
-mimc_bayesian {bayesian} ".format(bayesian="{bayesian}", TOL="{TOL}",
                                   dim=dim, seed="{seed}",
                                   gamma=" ".join([str(int(3/dim))]*dim),
                                   w=" ".join(["2"]*dim),
                                   s=" ".join(["4"]*dim),
                                   beta=" ".join(["2"]*dim))

cmd_multi = "python " + base + "-mimc_verbose False -db True -db_tag {tag} " + " -db_host {} ".format(args.db_host)
cmd_single = "python " + base + " -mimc_verbose True -db False "

if not args.multi:
    print(cmd_single.format(seed=0, bayesian=False, TOL=0.001, dim=args.mimc_dim))
else:
    realizations = 35
    TOLs = 0.1*np.sqrt(2.)**-np.arange(0., 16.)
    for TOL in TOLs:
        for i in range(0, realizations):
            print cmd_multi.format(bayesian=False, dim=args.mimc_dim,
                                   tag="PDE_dim{}".format(dim), TOL=TOL,
                                   seed=np.random.randint(2**32-1))
