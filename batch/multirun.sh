#!/bin/bash
for VAR in {1..10}
do
echo "Processing run $VAR"
python -u optim.py optjob_files/hypothermia_475.optjob ../data/clean_hypothermia/.csv > >(tee -a ../data/optimisation_$VAR.log)
done
