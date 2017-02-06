#!/bin/bash
# basic script to run a series of optimisation jobs for Tracy's paper
python optim.py -D lwp188_rocc.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_rocc.out
python optim.py -D lwp188_df.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_df.out
python optim.py -D lwp188_kunc.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_kunc.out
python optim.py -D lwp188_df_rocc.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_df_rocc.out
python optim.py -D lwp188_df_rocc_kunc.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_df_rocc_kunc.out

