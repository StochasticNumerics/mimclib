#!/bin/bash

# ./miproj_run.sh run sf-kink   | parallel -j28

#./miproj_run.sh run sf-matern | parallel -j28
{ ./miproj_run.sh run sf-kink & ./miproj_run.sh run sf-matern ; } | parallel -j28

./miproj_est.sh | parallel -j20

./plot_miproj_paper.sh | parallel -j20
