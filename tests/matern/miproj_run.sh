#!/bin/bash

# make
# rm -f data.sql

COMMON="python miproj_run.py -mimc_TOL 1e-10 -qoi_seed 0 -qoi_problem 0 -qoi_sigma 0.2  \
       -qoi_x0 0.3 0.4 0.6 -ksp_rtol 1e-25 -ksp_type gmres -qoi_a0 0 -qoi_f0 1 \
       -qoi_scale 10 -qoi_df_sig 0.5 -mimc_beta 2 -mimc_gamma 1 -mimc_h0inv 3  -mimc_verbose 1 \
       -db True -db_engine sqlite -db_name data.sql"

function run_cmd {
    # d max_lvl nu set extra_args
    echo $COMMON -mimc_max_lvl $3 -mimc_min_dim $2 -qoi_dim $2 -qoi_df_nu $4 \
         -db_tag $1-$2-$4-$5 -miproj_reuse_samples True \
         -miproj_pts_sampler $5 ${@:6}
}

function plot_cmd {
    echo ../plot_prog.py -db_engine sqlite -db_name data.sql -db_tag $1 \
         -o output/$1.pdf -verbose True -all_itr True \
         -qoi_exact_tag $1  &
}

run_cmd fixproj 1 5 3.5 optimal -mimc_min_dim 0
run_cmd fixproj 1 5 2.5 optimal -mimc_min_dim 0
run_cmd fixproj 1 5 1.5 optimal -mimc_min_dim 0
run_cmd fixproj 1 5 1.0 optimal -mimc_min_dim 0

run_cmd fixproj 1 5 3.5 arcsine -mimc_min_dim 0
run_cmd fixproj 1 5 2.5 arcsine -mimc_min_dim 0
run_cmd fixproj 1 5 1.5 arcsine -mimc_min_dim 0
run_cmd fixproj 1 5 1.0 arcsine -mimc_min_dim 0


run_cmd miproj 1 10 1.0 optimal
run_cmd miproj 1 10 1.5 optimal
run_cmd miproj 1 10 2.5 optimal
run_cmd miproj 1 10 3.5 optimal

run_cmd miproj 1 10 1.0 arcsine
run_cmd miproj 1 10 1.5 arcsine
run_cmd miproj 1 10 2.5 arcsine
run_cmd miproj 1 10 3.5 arcsine


run_cmd fixproj 3 4 4.5 optimal -mimc_min_dim 0
run_cmd fixproj 3 4 3.0 optimal -mimc_min_dim 0
run_cmd fixproj 3 4 4.5 arcsine -mimc_min_dim 0
run_cmd fixproj 3 4 3.0 arcsine -mimc_min_dim 0

run_cmd miproj 3 7 3.0 optimal
run_cmd miproj 3 7 4.5 optimal
run_cmd miproj 3 7 3.0 arcsine
run_cmd miproj 3 7 4.5 arcsine

# plot_cmd miproj1_1.0
# plot_cmd miproj1_1.5
# plot_cmd miproj1_2.5
# plot_cmd miproj1_3.5
# plot_cmd miproj3_4.5
# plot_cmd miproj3_3.0

# plot_cmd miproj1_20
# plot_cmd miproj1_10
# plot_cmd miproj1_20
# plot_cmd miproj1_4.5
