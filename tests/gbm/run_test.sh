#!/bin/bash

BAYESIAN="-mimc_bayesian False"

ipython run.py --  -mimc_TOL 0.001 -mimc_verbose True  \
        -qoi_sigma 0.1 -qoi_mu 1 -qoi_seed 0 \
        -mimc_dim 1 -mimc_w 1 -mimc_s 1 -mimc_gamma 1 -mimc_beta 2 $MIMC
