#!/usr/bin/python
import numpy as np
import argparse

'''
A script to generate a command for doing a MIMC run.

Usage: ./echo_test_cmd.py
prints the command.

To run a single mimc run, type
./echo_test_cmd.py |bash

Or multiple in parallel:
./echo_test_cmd.py -multi |parallel j x
where x is the number of processes to run in parallel.

'''


parser = argparse.ArgumentParser(add_help=True)
parser.register('type', 'bool',
                lambda v: v.lower() in ("yes", "true", "t", "1"))
parser.add_argument("-multi", type="bool", action="store",
                    default=False, help="True output a single command")
parser.add_argument("-db_host", type=str, action="store",
                    default="localhost",
                    help="True output a single command")
parser.add_argument("-TOL_max", type=float, action="store",
                    default=0.001,
                    help="True output a single command")
parser.add_argument("-TOL_min", type=float, action="store",
                    default=0.00001,
                    help="True output a single command")
parser.add_argument("-TOL_N", type=int, action="store",
                    default=8,
                    help="True output a single command")
parser.add_argument("-N_iter", type=int, action="store",
                    default=35,
                    help="True output a single command")



args = parser.parse_known_args()[0]

# Note that 0.6931 is np.log(2)
base = "run.py -mimc_TOL {TOL} -mimc_max_TOL 0.5  \
-qoi_seed {seed} \
-mimc_dim 2 -mimc_w 2 2 -mimc_s 0.5 1 -mimc_gamma 1 1 -mimc_beta 2 2 \
-mimc_bayesian {bayesian} "

cmd_multi = "python " + base + "-mimc_verbose False -db True -db_tag {tag} " + " -db_host {} ".format(args.db_host)
cmd_single = "python " + base + " -mimc_verbose True -db False "

if not args.multi:
    print(cmd_single.format(seed=0, bayesian=False, TOL=args.TOL_min))
else:
    realizations = args.N_iter # number of runs for each tolerance
    TOL_max = args.TOL_max
    TOL_min = args.TOL_min
    TOL_N = args.TOL_N
    TOLs = np.logspace(np.log10(TOL_max),np.log10(TOL_min),TOL_N)
    for TOL in TOLs:
        for i in range(0, realizations):
            print cmd_multi.format(bayesian=False, tag="HJM_Example_2_newnewrate", TOL=TOL,
                                   seed=np.random.randint(2**32-1))
