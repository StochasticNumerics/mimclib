#!/bin/bash

# ./miproj_run.sh run sf-kink   | parallel -j28

#./miproj_run.sh run sf-matern | parallel -j28

./miproj_run.sh run | parallel -j28

./plot_miproj_paper.sh | parallel -j20
