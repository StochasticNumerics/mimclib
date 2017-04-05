#!/bin/bash

./miproj_run.sh run | parallel -j20

./miproj_est.sh | parallel -j20

wait
