#!/usr/bin/python
cmd = "python run.py -mimc_TOL {TOL} -mimc_max_TOL 0.5 -mimc_verbose False  \
-qoi_sigma 0.1 -qoi_mu 1 -qoi_seed {seed} \
-mimc_dim 1 -mimc_w 1 -mimc_s 1 -mimc_gamma 1 -mimc_beta 2 \
-db True -db_user abdo -db_host 10.68.170.245 -db_tag {tag} \
-mimc_bayesian {bayesian}  "

import numpy as np
realizations = 35
TOLs = 0.1*2.**-np.arange(0., 10.)
for TOL in TOLs:
    for i in range(0, realizations):
        print cmd.format(bayesian=True, tag="GBM_bayes", TOL=TOL,
                         seed=np.random.randint(2**32-1))
        print cmd.format(bayesian=False, tag="GBM", TOL=TOL,
                         seed=np.random.randint(2**32-1))
