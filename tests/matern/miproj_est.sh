#!/bin/bash

EST_CMD="python miproj_esterr.py -db_engine mysql -db_name mimc -db_host 129.67.187.118"

echo $EST_CMD -db_tag "matern_reuse-%10.5%" -qoi_exact_tag matern-reuse-1-10.5
echo $EST_CMD -db_tag "matern_reuse-%8.5%" -qoi_exact_tag matern-reuse-1-8.5
echo $EST_CMD -db_tag "matern_reuse-%6.5%" -qoi_exact_tag matern-reuse-1-6.5
echo $EST_CMD -db_tag "matern_reuse-%4.5%" -qoi_exact_tag matern-reuse-1-4.5
echo $EST_CMD -db_tag "matern_reuse-%3.5%" -qoi_exact_tag matern-reuse-1-3.5
echo $EST_CMD -db_tag "matern_reuse-%2.5%" -qoi_exact_tag matern-reuse-1-2.5

echo $EST_CMD -db_tag "sf-kink-2-1%" -qoi_exact_tag sf-kink-2-1
echo $EST_CMD -db_tag "sf-kink-2-2%" -qoi_exact_tag sf-kink-2-2
