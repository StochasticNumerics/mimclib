#!/usr/bin/python
# Note that 0.6931 is np.log(2)

cmd = "python run.py -mimc_TOL {TOL} -mimc_max_TOL 0.5 -mimc_verbose False  \
-qoi_sigma 0.1 -qoi_mu 1 -qoi_seed {seed} \
-mimc_dim 1 -mimc_w 0.6931 -mimc_s 0.6931 -mimc_gamma 0.6931 -mimc_beta 2 \
-db True -db_tag {tag} \
-mimc_bayesian {bayesian}  "

import numpy as np
realizations = 35
TOLs = 0.1*2.**-np.arange(0., 15.)
for TOL in TOLs:
    for i in range(0, realizations):
        print cmd.format(bayesian=True, tag="GBM_bayes", TOL=TOL,
                         seed=np.random.randint(2**32-1))
        print cmd.format(bayesian=False, tag="GBM", TOL=TOL,
                         seed=np.random.randint(2**32-1))
