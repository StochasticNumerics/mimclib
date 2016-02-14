#!/bin/bash

MIMC="-mimc_dim 3"
MIMC="-mimc_dim 1 -mimc_bayesian True"

ipython run.py --  -mimc_TOL 0.001 -mimc_verbose True  \
        -qoi_dim 3 -qoi_x0 0.3,0.2,0.6 -qoi_sigma 0.16 -qoi_scale 1 \
        -ksp_rtol 1e-25 -ksp_type gmres \
        -mimc_w 2 -mimc_s 4 -mimc_gamma 1 -mimc_beta 2 $MIMC
