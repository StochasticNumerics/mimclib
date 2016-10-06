#!/usr/bin/python
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.register('type', 'bool',
                    lambda v: v.lower() in ("yes", "true", "t", "1", "y"))
    parser.add_argument("-tries", type=int, action="store",
                        default=0, help="Number of realizations")
    parser.add_argument("-db_tag", type=str, action="store",
                        default="GBM_testcase", help="Database tag")
    parser.add_argument("-mimc_bayesian", type="bool", action="store",
                        default=False)

    args, unknowns = parser.parse_known_args()

    base = "mimc_run.py -mimc_TOL {TOL} -mimc_max_TOL 0.5  \
    -qoi_sigma 0.1 -qoi_mu 1 -qoi_seed {seed} -mimc_moments 4 -mimc_M0 100 \
    -mimc_min_dim 1 -mimc_w 1 -mimc_s 1 -mimc_gamma 1 -mimc_beta 2 \
    -mimc_confidence 0.95 -mimc_theta 0.5 -mimc_bayes_fit_lvls 5 \
    -mimc_bayes_k1 0.01 -mimc_reuse_samples True "

    if args.mimc_bayesian:
        base += " -mimc_bayesian True -mimc_theta 0.2 "
    else:
        base += " -mimc_bayesian False -mimc_theta 0.5 "

    base += " ".join(unknowns)

    if args.tries == 0:
        cmd_single = "python " + base + " -mimc_verbose 10 -db False -db_tag {tag} "
        print(cmd_single.format(seed=0, TOL=0.001, tag=args.db_tag))
    else:
        cmd_multi = "python " + base + \
                    " -mimc_verbose 0 -db True -db_tag {tag} "
        #TOLs = 0.05*np.sqrt(2.)**-np.arange(0., 21.)
        TOL = 0.05*np.sqrt(2.)**-18
        for i in range(0, args.tries):
            print cmd_multi.format(tag=args.db_tag, TOL=TOL,
                                       seed=np.random.randint(2**32-1))
