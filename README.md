# MIMClib

MIMClib is open source library for UQ problems written mainly in Python.
The library aims to give the user an easy to use, customizable and extendable
library to solve a wide class of stochastic problems subject to 
error tolerance efficiently.

## Installing

### On a generic GNU/Linux system

1. Install Dependencies: python-pip mysql-server mysql-client libmysqlclient-dev build-essential ipython libpng-dev libfreetype6-dev libxft-dev libpython-dev liblapack-dev libblas-dev gfortran parallel numpy matplotlib scipy mysql-python (Package names may be slightly different in your particular platform)
2. Clone this repository `https://github.com/StochasticNumerics/mimclib.git`
3. In the downloaded folder run `make` and `make pip`
4. Create the database `python -c 'from mimclib.db import MIMCDatabase; print MIMCDatabase().DBCreationScript();' | mysql -u root -p`
5. If you don't have a mySQL user, create one and give it appropriate privileges `echo -e "CREATE USER '$USER'@'%';\nGRANT ALL PRIVILEGES ON mimc.* TO '$USER'@'%' WITH GRANT OPTION;" | mysql -u root -p`

## Running the GBM Example

### Single run without the mySQL backend

In the directory
[tests/gbm](https://github.com/StochasticNumerics/mimclib/tree/master/tests/gbm),
run `./single_run_example`

### Parallel runs, storing the results into mySQL

In the directory
[tests/gbm](https://github.com/StochasticNumerics/mimclib/tree/master/tests/gbm)
run `./parallel_run_example`

This generates run commands using
[echo_test_cmd.py](https://github.com/StochasticNumerics/mimclib/blob/master/tests/gbm/echo_test_cmd.py),
and pipelines it to
[parallel](https://www.gnu.org/software/parallel/).
Each of the commands will
have a separate random seed and parallel will run them
asynchronously storing the results in the database with
the database identifier tag Parallelrun_example.

There are multiple additional tags that can be passed to
[./mimc_run.py](https://github.com/StochasticNumerics/mimclib/blob/master/tests/gbm/mimc_run.py),
including the initial value of the geometric
Brownian walk, final time and others. Feel free to experiment
with these, but if you do, remember to change the database
identifier tag db_tag from Parallelrun_example. Failure
to do so will result in simulations with different parameters
being stored in the database with identical tags.

## Running your own examples

Following the example in
[tests/gbm](https://github.com/StochasticNumerics/mimclib/tree/master/tests/gbm),
you can copy the example directory
and generate your own ideas based on that. You can replace the 
function
[mySampleQoI](https://github.com/StochasticNumerics/mimclib/blob/master/tests/gbm/mimc_run.py#L65)
in
[mimc_run.py](https://github.com/StochasticNumerics/mimclib/blob/master/tests/gbm/mimc_run.py)
with your own implementation.

Your implementation needs to return a numpy array with dimensions
Mxl where M is the input parameter describing the sample size
and l is the length of indices. The ith element on the jth row
of the array returned will need to contain the quantity of interest
at the discretisation level of the jth element of the multi-index
corresponding to the jth element of the input parameter inds. All
elements in the same column of the array will need to correspond to
the same element in the probability space.

Naturally, if you are not the geometric Brownian motion solver, you can remove
the compiled version of the code in
[mimc_run.py](https://github.com/StochasticNumerics/mimclib/blob/master/tests/gbm/mimc_run.py#L41)

If your problem features convergence rates different from the
geometric brownian walk, remember to adjust these accordingly in
your copy of
[echo_test_cmd.py](https://github.com/StochasticNumerics/mimclib/blob/master/tests/gbm/echo_test_cmd.py).
The most relevant command line parameters here are
[mimc_dim 1, mimc_w 1, mimc_s 1, mimc_gamma and mimc_beta.](https://github.com/StochasticNumerics/mimclib/blob/master/tests/gbm/echo_test_cmd.py#L18)


## Plotting

The essential plotting routine for mimclib can be found at
[tests/plot_data.py](https://github.com/StochasticNumerics/mimclib/blob/master/tests/plot_data.py).
This script generates a booklet of different plots describing the convergence and other
characteristics of a mimc run.

A flavour of how to use plot_data.py for the runs generated in the
[GBM problem](https://github.com/StochasticNumerics/mimclib#parallel-runs-storing-the-results-into-mysql),
one may run the command
`./plot_example` in the directory tests/gbm.

## License

Code in this repository is released under
[GNU General Public License 2.0](https://github.com/StochasticNumerics/mimclib/blob/master/LICENSE)

## References

[Multi Index Monte Carlo: When Sparsity Meets Sampling](http://link.springer.com/article/10.1007/s00211-015-0734-5)



