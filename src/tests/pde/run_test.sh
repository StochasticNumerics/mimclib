#!/bin/bash

ipython run.py -- -mimc_dim 1 -mimc_TOL 0.001 -mimc_verbose True -mimc_gamma 1 \
-qoi_dim 3 -qoi_x0 0.3,0.2,0.6 -qoi_sigma 0.16 -qoi_scale 1 \
        -ksp_rtol 1e-25 -ksp_type gmres\
        -mimc_bayesian True -mimc_w 2 -mimc_s 4
