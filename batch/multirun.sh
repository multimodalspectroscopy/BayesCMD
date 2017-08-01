#!/bin/bash
for VAR in {1..10}
do
echo "Processing run $VAR"
python -u optim.py optjob_files/hx01_BS.optjob ../data/hx01.csv > >(tee -a ../data/optimisation_$VAR.log)
done
