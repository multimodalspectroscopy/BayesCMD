#!/bin/bash
for VAR in {1..10}
do
echo "Processing run $VAR"
python -u optim.py optjob_files/hypothermia_475.optjob ../data/clean_hypothermia/cleaned_LWP475_filtered.csv > >(tee -a ../data/optimisation_$VAR.log)
done
