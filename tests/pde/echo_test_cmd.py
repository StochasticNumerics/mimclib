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


# MIMC="-mimc_dim 3"
# MIMC="-mimc_dim 1 -mimc_bayesian True"

# python run.py -mimc_TOL 0.001 -mimc_verbose True -qoi_dim 3 -qoi_x0 0.3,0.2,0.6 -qoi_sigma 0.16 -qoi_scale 1 -ksp_rtol 1e-25 -ksp_type gmres -mimc_w 2 -mimc_s 4 -mimc_gamma 1 -mimc_beta 2 $MIMC

# Note that 0.6931 is np.log(2)
base = "run.py -mimc_TOL {TOL} -mimc_dim 3 -qoi_seed {seed} -qoi_dim 3 \
-qoi_x0 0.3,0.2,0.6 -qoi_sigma 0.16 -qoi_scale 1 -ksp_rtol 1e-25 \
-ksp_type gmres -mimc_w 2 2 2 -mimc_s 4 4 4 -mimc_gamma 2 2 2 -mimc_beta 2 2 2 \
-mimc_bayesian {bayesian} "

cmd_multi = "python " + base + "-mimc_verbose False -db True -db_tag {tag} " + " -db_host {} ".format(args.db_host)
cmd_single = "python " + base + " -mimc_verbose True -db False "

if not args.multi:
    print(cmd_single.format(seed=0, bayesian=False, TOL=0.01))
else:
    realizations = 35
    TOLs = 0.1*np.sqrt(2.)**-np.arange(0., 10.)
    for TOL in TOLs:
        for i in range(0, realizations):
            print cmd_multi.format(bayesian=False, tag="PDE", TOL=TOL,
                                   seed=np.random.randint(2**32-1))
