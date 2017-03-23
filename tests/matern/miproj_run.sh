#!/bin/bash

# make
# rm -f data.sql

HOST='129.67.187.118'
BASETAG='re-'
#BASETAG=''
COMMON="-mimc_TOL 1e-10 -qoi_seed 0 -qoi_problem 0 -qoi_sigma 0.2  \
       -qoi_x0 0.3 0.4 0.6 -ksp_rtol 1e-25 -ksp_type gmres -qoi_a0 0 -qoi_f0 1 \
       -qoi_scale 10 -qoi_df_sig 0.5 -mimc_beta 2 -mimc_gamma 1 -mimc_h0inv 3  -mimc_verbose 1 \
       -qoi_set_adaptive False -qoi_set_xi 2 -qoi_set_dexp 2.08 \
       -miproj_reuse_samples False -db True -db_name mimc -db_host $HOST "


COMMON_EST="-mimc_TOL 1e-10 -qoi_seed 0 -qoi_problem 0 -qoi_sigma 0.2  \
       -qoi_x0 0.3 0.4 0.6 -ksp_rtol 1e-25 -ksp_type gmres -qoi_a0 0 -qoi_f0 1 \
       -qoi_scale 10 -qoi_df_sig 0.5 -mimc_beta 2 -mimc_gamma 1 -mimc_h0inv 3  -mimc_verbose 1 \
       -qoi_set_adaptive False -qoi_set_xi 2 -qoi_set_dexp 2.08 \
       -miproj_reuse_samples False -db_name mimc -db_host $HOST "

function run_cmd {
    # d max_lvl nu set extra_args
    echo python miproj_run.py $COMMON -mimc_max_lvl $3 -mimc_min_dim $2 \
         -qoi_dim $2 -qoi_df_nu $4 -db_tag $BASETAG$1-$2-$4-optimal \
         -miproj_pts_sampler optimal ${@:5}

    echo python miproj_run.py $COMMON -mimc_max_lvl $3 -mimc_min_dim $2 \
         -qoi_dim $2 -qoi_df_nu $4 -db_tag $BASETAG$1-$2-$4-arcsine \
         -miproj_pts_sampler arcsine ${@:5}
}

function plot_cmd {
    echo ../plot_prog.py -db_engine mysql -db_name mimc -db_host $HOST \
         -db_tag $BASETAG$1-optimal -o output/$BASETAG$1-optimal.pdf -verbose True -all_itr True \
         -qoi_exact_tag $1-optimal

    echo ../plot_prog.py -db_engine mysql -db_name mimc -db_host $HOST \
         -db_tag $BASETAG$1-arcsine -o output/$BASETAG$1-arcsine.pdf -verbose True -all_itr True \
         -qoi_exact_tag $1-arcsine

    echo ./plot_cond.py -db_name mimc -db_host $HOST -db_tag $BASETAG$1-optimal
    echo ./plot_cond.py -db_name mimc -db_host $HOST -db_tag $BASETAG$1-arcsine
}


function errest_cmd {
    echo python miproj_esterr.py $COMMON_EST -mimc_max_lvl $3 -mimc_min_dim $2 -qoi_dim $2 -qoi_df_nu $4 \
         -db_tag $BASETAG$1-$2-$4-optimal \
         -miproj_pts_sampler optimal ${@:5}

    echo python miproj_esterr.py $COMMON_EST -mimc_max_lvl $3 -mimc_min_dim $2 -qoi_dim $2 -qoi_df_nu $4 \
         -db_tag $BASETAG$1-$2-$4-arcsine \
         -miproj_pts_sampler arcsine ${@:5}
}

if [ "$1" = "plot" ]; then
    # plot_cmd fixproj-1-1.0
    # plot_cmd fixproj-1-1.5
    plot_cmd fixproj-1-2.5
    plot_cmd fixproj-1-3.5
    plot_cmd fixproj-1-4.5
    plot_cmd fixproj-1-6.5

    # plot_cmd miproj-1-1.0
    # plot_cmd miproj-1-1.5
    plot_cmd miproj-1-2.5
    plot_cmd miproj-1-3.5
    plot_cmd miproj-1-4.5
    plot_cmd miproj-1-6.5

    # plot_cmd miproj-3-4.5
    # plot_cmd miproj-3-3.0
elif [ "$1" = "est" ]; then
    errest_cmd fixproj 1 10 6.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 6.99
    errest_cmd fixproj 1 10 4.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 4.99
    errest_cmd fixproj 1 10 3.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 3.99
    errest_cmd fixproj 1 10 2.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 2.99

    errest_cmd miproj 1 10 6.5 -miproj_min_dim 5  -qoi_set_sexp 6.99
    errest_cmd miproj 1 10 4.5 -miproj_min_dim 5  -qoi_set_sexp 4.99
    errest_cmd miproj 1 10 3.5 -miproj_min_dim 5  -qoi_set_sexp 3.99
    errest_cmd miproj 1 10 2.5 -miproj_min_dim 5  -qoi_set_sexp 2.99
else
    run_cmd fixproj 1 10 6.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 6.99
    run_cmd fixproj 1 10 4.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 4.99
    run_cmd fixproj 1 10 3.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 3.99
    run_cmd fixproj 1 10 2.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 2.99
    # run_cmd fixproj 1 10 1.5 -mimc_min_dim 0 -miproj_min_dim 5
    # run_cmd fixproj 1 10 1.0 -mimc_min_dim 0 -miproj_min_dim 5

    run_cmd miproj 1 10 6.5 -miproj_min_dim 5  -qoi_set_sexp 6.99
    run_cmd miproj 1 10 4.5 -miproj_min_dim 5  -qoi_set_sexp 4.99
    run_cmd miproj 1 10 3.5 -miproj_min_dim 5  -qoi_set_sexp 3.99
    run_cmd miproj 1 10 2.5 -miproj_min_dim 5  -qoi_set_sexp 2.99

    # run_cmd miproj 1 10 1.5 -miproj_min_dim 5
    # run_cmd miproj 1 10 1.0 -miproj_min_dim 5

    # run_cmd fixproj 3 7 3.0 -mimc_min_dim 0 -miproj_min_dim 5
    # run_cmd fixproj 3 7 4.5 -mimc_min_dim 0 -miproj_min_dim 5

    # run_cmd miproj 3 7 3.0 -miproj_min_dim 5
    # run_cmd miproj 3 7 4.5 -miproj_min_dim 5
fi
