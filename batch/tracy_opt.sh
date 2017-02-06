#!/bin/bash
# basic script to run a series of optimisation jobs for Tracy's paper
python optim.py -D lwp180_CCO.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_CCO.out
python optim.py -D lwp180_HHb.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_HHb.out
python optim.py -D lwp180_P.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_P.out
python optim.py -D lwp180_pH.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_pH.out

tail -n 2 lwp180_CCO.out > lwp180_all.out
tail -n 3 lwp180_HHb.out >> lwp180_all.out
tail -n 3 lwp180_P.out >> lwp180_all.out
tail -n 1 lwp180_pH.out >> lwp180_all.out

python optim.py -D lwp188_CCO.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_CCO.out
python optim.py -D lwp188_HHb.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_HHb.out
python optim.py -D lwp188_P.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_P.out
python optim.py -D lwp188_pH.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_pH.out

tail -n 2 lwp188_CCO.out > lwp188_all.out
tail -n 3 lwp188_HHb.out >> lwp188_all.out
tail -n 3 lwp188_P.out >> lwp188_all.out
tail -n 1 lwp188_pH.out >> lwp188_all.out
