#!/bin/bash

# make
# rm -f data.sql
PROBLEM='matern'
DB_CONN='-db_engine mysql -db_name mimc -db_host 129.67.187.118 '
BASETAG="$PROBLEM-reuse-"
COMMON="-qoi_seed 0 -qoi_sigma 0.2  \
       -qoi_x0 0.3 0.4 0.6 -ksp_rtol 1e-25 -ksp_type gmres -qoi_a0 0 -qoi_f0 1 \
       -qoi_scale 10 -qoi_df_sig 0.5 -mimc_beta 2 -mimc_gamma 1 -mimc_h0inv 3  \
       -mimc_verbose 1 -qoi_set_xi 2 -qoi_set_dexp 2.08 \
       $DB_CONN "
EST_CMD="python miproj_esterr.py $COMMON "
RUN_CMD="OPENBLAS_NUM_THREADS=1 python miproj_run.py -qoi_problem $PROBLEM \
       -mimc_TOL 0 -qoi_seed 0 -mimc_beta 2 -mimc_gamma 1 \
       -qoi_set_xi 2 -qoi_set_dexp 2.08 \
       -miproj_reuse_samples True -db True $COMMON "

function run_cmd {
    # d max_lvl nu set extra_args
    echo  $RUN_CMD -mimc_max_lvl $3 -mimc_min_dim $2 \
          -qoi_dim $2 -qoi_df_nu $4 -db_tag $BASETAG$2-$4$1 \
          -miproj_pts_sampler optimal ${@:5}

    # echo python miproj_run.py $COMMON -mimc_max_lvl $3 -mimc_min_dim $2 \
    #      -qoi_dim $2 -qoi_df_nu $4 -db_tag $BASETAG$2-$4$1-arcsine \
    #      -miproj_pts_sampler arcsine ${@:5}
}

function plot_cmd {
    echo ../plot_prog.py $DB_CONN \
         -db_tag $BASETAG$2-$4$1 -o output/self-$BASETAG$2-$4$1.pdf \
         -verbose True -all_itr True -qoi_exact_tag $BASETAG$2-$4$1

    # echo ./plot_cond.py $DB_CONN -db_tag $BASETAG$1
    # echo ../plot_prog.py $DB_CONN \
    #      -db_tag $BASETAG$1-arcsine -o output/$BASETAG$1-arcsine.pdf \
    #      -verbose True -all_itr True #-qoi_exact_tag $BASETAG$1-arcsine
    #echo ./plot_cond.py $DB_CONN -db_tag $BASETAG$1-arcsine
}

function plotest_cmd {
    echo ../plot_prog.py "$DB_CONN" \
         -db_tag "$BASETAG$2-$4$1" -o "output/$BASETAG$2-$4$1.pdf" \
         -verbose True -all_itr True

    # echo ./plot_cond.py $DB_CONN -db_tag $BASETAG$1-optimal
    # echo ../plot_prog.py $DB_CONN \
    #      -db_tag $BASETAG$1-arcsine -o output/$BASETAG$1-arcsine.pdf \
    #      -verbose True -all_itr True #-qoi_exact_tag $BASETAG$1-arcsine
    #echo ./plot_cond.py $DB_CONN -db_tag $BASETAG$1-arcsine
}


function errest_cmd {
    echo $EST_CMD -mimc_max_lvl "$3" -mimc_min_dim "$2" \
         -mimc_fix_lvl 10 \
         -qoi_dim "$2" -qoi_df_nu "$4" -db_tag "$BASETAG$2-$4$1" \
         -miproj_pts_sampler optimal "${@:5}" \
         "; " ../plot_prog.py "$DB_CONN" \
         -db_tag "$BASETAG$2-$4$1" -o "output/$BASETAG$2-$4$1.pdf" \
         -verbose True -all_itr True

    # echo python miproj_esterr.py $COMMON_EST -mimc_max_lvl $3 -mimc_min_dim $2 -qoi_dim $2 -qoi_df_nu $4 \
    #      -db_tag $BASETAG$1-$2-$4-arcsine \
    #      -miproj_pts_sampler arcsine ${@:5}
}

function all_cmds {
    if [ "$1" = "plot" ]; then
        plot_cmd "${@:2}"
    elif [ "$1" = "est" ]; then
        errest_cmd ${@:2}
    elif [ "$1" = "plot_est" ]; then
        plotest_cmd "${@:2}"
    elif [ "$1" = "run" ]; then
        run_cmd "${@:2}"
    fi;
}

#all_cmds $1 adapt-finite 1 10 2.5 -mimc_min_dim 0 -miproj_min_dim 5 -miproj_max_dim 5
#all_cmds $1 finite 1 10 2.5 -mimc_min_dim 0 -miproj_min_dim 5 -qoi_set_sexp 1 -miproj_max_dim 5 -qoi_set_adaptive False

# all_cmds $1 adapt-fixproj 1 10 10.5 -mimc_min_dim 0 -miproj_min_dim 5
# all_cmds $1 adapt-fixproj 1 10 8.5 -mimc_min_dim 0 -miproj_min_dim 5
# all_cmds $1 adapt-fixproj 1 10 6.5 -mimc_min_dim 0 -miproj_min_dim 5
# all_cmds $1 adapt-fixproj 1 10 4.5 -mimc_min_dim 0 -miproj_min_dim 5
# all_cmds $1 adapt-fixproj 1 10 3.5 -mimc_min_dim 0 -miproj_min_dim 5
# all_cmds $1 adapt-fixproj 1 10 2.5 -mimc_min_dim 0 -miproj_min_dim 5

# all_cmds $1 fixproj 1 10 10.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 11. -qoi_set_adaptive False
# all_cmds $1 fixproj 1 10 8.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 9.0 -qoi_set_adaptive False
# all_cmds $1 fixproj 1 10 6.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 7.0 -qoi_set_adaptive False
# all_cmds $1 fixproj 1 10 4.5 -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 5.0 -qoi_set_adaptive False
# all_cmds $1 fixproj 1 10 3.5 -mimc_min_dim 0 -miproj_min_dim 5 -qoi_set_sexp 4.0 -qoi_set_adaptive False
# all_cmds $1 fixproj 1 10 2.5 -mimc_min_dim 0 -miproj_min_dim 5 -qoi_set_sexp 3.0 -qoi_set_adaptive False

all_cmds $1 -adapt 1 10 10.5 -miproj_min_dim 5
all_cmds $1 -adapt 1 10 8.5 -miproj_min_dim 5
all_cmds $1 -adapt 1 10 6.5 -miproj_min_dim 5
all_cmds $1 -adapt 1 10 4.5 -miproj_min_dim 5
all_cmds $1 -adapt 1 10 3.5 -miproj_min_dim 5
all_cmds $1 -adapt 1 10 2.5 -miproj_min_dim 5

all_cmds $1 "" 1 10 10.5 -miproj_min_dim 5 -qoi_set_sexp 11. -qoi_set_adaptive False
all_cmds $1 "" 1 10 8.5 -miproj_min_dim 5 -qoi_set_sexp 9. -qoi_set_adaptive False
all_cmds $1 "" 1 10 6.5 -miproj_min_dim 5 -qoi_set_sexp 7. -qoi_set_adaptive False
all_cmds $1 "" 1 10 4.5 -miproj_min_dim 5 -qoi_set_sexp 5. -qoi_set_adaptive False
all_cmds $1 "" 1 10 3.5 -miproj_min_dim 5 -qoi_set_sexp 4. -qoi_set_adaptive False
all_cmds $1 "" 1 10 2.5 -miproj_min_dim 5 -qoi_set_sexp 3. -qoi_set_adaptive False

for i in {0..8}
do
    #all_cmds $1 fix-$i 1 6 10.5 -miproj_fix_lvl $i -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 11. -qoi_set_adaptive False
    #all_cmds $1 fix-$i 1 6 8.5 -miproj_fix_lvl $i -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 9.0 -qoi_set_adaptive False
    all_cmds $1 -fix-$i 1 6 6.5 -miproj_fix_lvl $i -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 7.0 -qoi_set_adaptive False
    all_cmds $1 -fix-$i 1 6 4.5 -miproj_fix_lvl $i -mimc_min_dim 0 -miproj_min_dim 5  -qoi_set_sexp 5.0 -qoi_set_adaptive False
    all_cmds $1 -fix-$i 1 6 3.5 -miproj_fix_lvl $i -mimc_min_dim 0 -miproj_min_dim 5 -qoi_set_sexp 4.0 -qoi_set_adaptive False
    all_cmds $1 -fix-$i 1 6 2.5 -miproj_fix_lvl $i -mimc_min_dim 0 -miproj_min_dim 5 -qoi_set_sexp 3.0 -qoi_set_adaptive False
done
