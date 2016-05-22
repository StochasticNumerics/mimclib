
### Installation on a Debian system:

Command:
`wget https://raw.githubusercontent.com/StochasticNumerics/mimclib/master/mimc_install.sh && bash mimc_install.sh`


### TODO
0. This README
1. Move features and plots from old code (CMLMC and MIMC)
2. Implement vector quantities of interest
3. Logical checks on the convergence of the variance, work and
   expectation while MLMC is running and printing useful messages to
   indicate possible solutions. Also a check if TOL << QoI.
4. Computing and using the kurtosis when possible to give a confidence
   interval for the variance.
5. Implementing some form of parallelization support.
6. Adaptive construction of the MIMC set.
7. Implement a storing mechanism in files instead of database.
