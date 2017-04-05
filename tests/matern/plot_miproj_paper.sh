#!/bin/bash

# Rule: No arguments
DB_CONN='-db_engine mysql -db_name mimc -db_host 129.67.187.118 '

for nu in 2.5 3.5 4.5 6.5 8.5 10.5
do
    echo ./plot_miproj_paper.py $DB_CONN \
         -db_tag matern-1-$nu -o output/matern-$nu.pdf \
         -verbose True -all_itr True
done

echo ./plot_miproj_paper.py $DB_CONN \
     -db_tag poisson-kink-1 -o output/poisson-kink-1.pdf \
     -verbose True -all_itr True

echo ./plot_miproj_paper.py $DB_CONN \
     -db_tag poisson-kink-2 -o output/poisson-kink-2.pdf \
     -verbose True -all_itr True
