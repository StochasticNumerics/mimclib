#!/bin/bash

# Rule: No arguments
DB_CONN='-db_engine mysql -db_name mimc -db_host 129.67.187.118 '

# for nu in 2.5 3.5 4.5 6.5 8.5 10.5
# do
#     echo ./plot_miproj_paper.py $DB_CONN \
#          -db_tag sf-matern-1-$nu -o output/matern-$nu.pdf \
#          -verbose True -all_itr True
# done

for N in 2 3 4 6
do
    echo ./plot_miproj_paper.py $DB_CONN \
         -db_tag sf-kink-2-$N-adapt -o output/poisson-kink-$N \
         -verbose True -all_itr True # -qoi_exact_tag sf-kink-2-$N-adapt
done
