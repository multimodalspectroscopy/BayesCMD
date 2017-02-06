#!/bin/bash
# super-clunky script to run optimisations repeatedly for Tracy's paper
python optim.py -D lwp180_CCO.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_CCO.out
python optim.py -D lwp180_HHb.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_HHb.out
python optim.py -D lwp180_P.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_P.out
python optim.py -D lwp180_pH.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_pH.out

tail -n 2 lwp180_CCO.out > lwp180_all_x1.out
tail -n 3 lwp180_HHb.out >> lwp180_all_x1.out
tail -n 3 lwp180_P.out >> lwp180_all_x1.out
tail -n 1 lwp180_pH.out >> lwp180_all_x1.out

python optim.py -D lwp180_CCO.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_CCO.out
python optim.py -D lwp180_HHb.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_HHb.out
python optim.py -D lwp180_P.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_P.out
python optim.py -D lwp180_pH.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_pH.out

tail -n 2 lwp180_CCO.out > lwp180_all_x2.out
tail -n 3 lwp180_HHb.out >> lwp180_all_x2.out
tail -n 3 lwp180_P.out >> lwp180_all_x2.out
tail -n 1 lwp180_pH.out >> lwp180_all_x2.out

python optim.py -D lwp180_CCO.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_CCO.out
python optim.py -D lwp180_HHb.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_HHb.out
python optim.py -D lwp180_P.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_P.out
python optim.py -D lwp180_pH.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_pH.out

tail -n 2 lwp180_CCO.out > lwp180_all_x3.out
tail -n 3 lwp180_HHb.out >> lwp180_all_x3.out
tail -n 3 lwp180_P.out >> lwp180_all_x3.out
tail -n 1 lwp180_pH.out >> lwp180_all_x3.out

python optim.py -D lwp180_CCO.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_CCO.out
python optim.py -D lwp180_HHb.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_HHb.out
python optim.py -D lwp180_P.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_P.out
python optim.py -D lwp180_pH.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_pH.out

tail -n 2 lwp180_CCO.out > lwp180_all_x4.out
tail -n 3 lwp180_HHb.out >> lwp180_all_x4.out
tail -n 3 lwp180_P.out >> lwp180_all_x4.out
tail -n 1 lwp180_pH.out >> lwp180_all_x4.out

python optim.py -D lwp180_CCO.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_CCO.out
python optim.py -D lwp180_HHb.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_HHb.out
python optim.py -D lwp180_P.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_P.out
python optim.py -D lwp180_pH.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_pH.out

tail -n 2 lwp180_CCO.out > lwp180_all_x5.out
tail -n 3 lwp180_HHb.out >> lwp180_all_x5.out
tail -n 3 lwp180_P.out >> lwp180_all_x5.out
tail -n 1 lwp180_pH.out >> lwp180_all_x5.out

python optim.py -D lwp180_CCO.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_CCO.out
python optim.py -D lwp180_HHb.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_HHb.out
python optim.py -D lwp180_P.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_P.out
python optim.py -D lwp180_pH.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_pH.out

tail -n 2 lwp180_CCO.out > lwp180_all_x6.out
tail -n 3 lwp180_HHb.out >> lwp180_all_x6.out
tail -n 3 lwp180_P.out >> lwp180_all_x6.out
tail -n 1 lwp180_pH.out >> lwp180_all_x6.out

python optim.py -D lwp180_CCO.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_CCO.out
python optim.py -D lwp180_HHb.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_HHb.out
python optim.py -D lwp180_P.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_P.out
python optim.py -D lwp180_pH.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_pH.out

tail -n 2 lwp180_CCO.out > lwp180_all_x7.out
tail -n 3 lwp180_HHb.out >> lwp180_all_x7.out
tail -n 3 lwp180_P.out >> lwp180_all_x7.out
tail -n 1 lwp180_pH.out >> lwp180_all_x7.out

python optim.py -D lwp180_CCO.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_CCO.out
python optim.py -D lwp180_HHb.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_HHb.out
python optim.py -D lwp180_P.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_P.out
python optim.py -D lwp180_pH.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_pH.out

tail -n 2 lwp180_CCO.out > lwp180_all_x8.out
tail -n 3 lwp180_HHb.out >> lwp180_all_x8.out
tail -n 3 lwp180_P.out >> lwp180_all_x8.out
tail -n 1 lwp180_pH.out >> lwp180_all_x8.out

python optim.py -D lwp180_CCO.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_CCO.out
python optim.py -D lwp180_HHb.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_HHb.out
python optim.py -D lwp180_P.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_P.out
python optim.py -D lwp180_pH.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_pH.out

tail -n 2 lwp180_CCO.out > lwp180_all_x9.out
tail -n 3 lwp180_HHb.out >> lwp180_all_x9.out
tail -n 3 lwp180_P.out >> lwp180_all_x9.out
tail -n 1 lwp180_pH.out >> lwp180_all_x9.out

python optim.py -D lwp180_CCO.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_CCO.out
python optim.py -D lwp180_HHb.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_HHb.out
python optim.py -D lwp180_P.optjob scratch/lwp/lwp180_p_fix_pre.csv | tee lwp180_P.out
python optim.py -D lwp180_pH.optjob scratch/lwp/lwp180_p_fix.csv | tee lwp180_pH.out

tail -n 2 lwp180_CCO.out > lwp180_all_x0.out
tail -n 3 lwp180_HHb.out >> lwp180_all_x0.out
tail -n 3 lwp180_P.out >> lwp180_all_x0.out
tail -n 1 lwp180_pH.out >> lwp180_all_x0.out

python optim.py -D lwp188_CCO.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_CCO.out
python optim.py -D lwp188_HHb.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_HHb.out
python optim.py -D lwp188_P.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_P.out
python optim.py -D lwp188_pH.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_pH.out

tail -n 2 lwp188_CCO.out > lwp188_all_x1.out
tail -n 3 lwp188_HHb.out >> lwp188_all_x1.out
tail -n 3 lwp188_P.out >> lwp188_all_x1.out
tail -n 1 lwp188_pH.out >> lwp188_all_x1.out

python optim.py -D lwp188_CCO.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_CCO.out
python optim.py -D lwp188_HHb.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_HHb.out
python optim.py -D lwp188_P.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_P.out
python optim.py -D lwp188_pH.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_pH.out

tail -n 2 lwp188_CCO.out > lwp188_all_x2.out
tail -n 3 lwp188_HHb.out >> lwp188_all_x2.out
tail -n 3 lwp188_P.out >> lwp188_all_x2.out
tail -n 1 lwp188_pH.out >> lwp188_all_x2.out

python optim.py -D lwp188_CCO.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_CCO.out
python optim.py -D lwp188_HHb.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_HHb.out
python optim.py -D lwp188_P.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_P.out
python optim.py -D lwp188_pH.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_pH.out

tail -n 2 lwp188_CCO.out > lwp188_all_x3.out
tail -n 3 lwp188_HHb.out >> lwp188_all_x3.out
tail -n 3 lwp188_P.out >> lwp188_all_x3.out
tail -n 1 lwp188_pH.out >> lwp188_all_x3.out

python optim.py -D lwp188_CCO.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_CCO.out
python optim.py -D lwp188_HHb.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_HHb.out
python optim.py -D lwp188_P.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_P.out
python optim.py -D lwp188_pH.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_pH.out

tail -n 2 lwp188_CCO.out > lwp188_all_x4.out
tail -n 3 lwp188_HHb.out >> lwp188_all_x4.out
tail -n 3 lwp188_P.out >> lwp188_all_x4.out
tail -n 1 lwp188_pH.out >> lwp188_all_x4.out

python optim.py -D lwp188_CCO.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_CCO.out
python optim.py -D lwp188_HHb.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_HHb.out
python optim.py -D lwp188_P.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_P.out
python optim.py -D lwp188_pH.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_pH.out

tail -n 2 lwp188_CCO.out > lwp188_all_x5.out
tail -n 3 lwp188_HHb.out >> lwp188_all_x5.out
tail -n 3 lwp188_P.out >> lwp188_all_x5.out
tail -n 1 lwp188_pH.out >> lwp188_all_x5.out

python optim.py -D lwp188_CCO.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_CCO.out
python optim.py -D lwp188_HHb.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_HHb.out
python optim.py -D lwp188_P.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_P.out
python optim.py -D lwp188_pH.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_pH.out

tail -n 2 lwp188_CCO.out > lwp188_all_x6.out
tail -n 3 lwp188_HHb.out >> lwp188_all_x6.out
tail -n 3 lwp188_P.out >> lwp188_all_x6.out
tail -n 1 lwp188_pH.out >> lwp188_all_x6.out

python optim.py -D lwp188_CCO.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_CCO.out
python optim.py -D lwp188_HHb.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_HHb.out
python optim.py -D lwp188_P.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_P.out
python optim.py -D lwp188_pH.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_pH.out

tail -n 2 lwp188_CCO.out > lwp188_all_x7.out
tail -n 3 lwp188_HHb.out >> lwp188_all_x7.out
tail -n 3 lwp188_P.out >> lwp188_all_x7.out
tail -n 1 lwp188_pH.out >> lwp188_all_x7.out

python optim.py -D lwp188_CCO.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_CCO.out
python optim.py -D lwp188_HHb.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_HHb.out
python optim.py -D lwp188_P.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_P.out
python optim.py -D lwp188_pH.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_pH.out

tail -n 2 lwp188_CCO.out > lwp188_all_x8.out
tail -n 3 lwp188_HHb.out >> lwp188_all_x8.out
tail -n 3 lwp188_P.out >> lwp188_all_x8.out
tail -n 1 lwp188_pH.out >> lwp188_all_x8.out

python optim.py -D lwp188_CCO.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_CCO.out
python optim.py -D lwp188_HHb.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_HHb.out
python optim.py -D lwp188_P.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_P.out
python optim.py -D lwp188_pH.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_pH.out

tail -n 2 lwp188_CCO.out > lwp188_all_x9.out
tail -n 3 lwp188_HHb.out >> lwp188_all_x9.out
tail -n 3 lwp188_P.out >> lwp188_all_x9.out
tail -n 1 lwp188_pH.out >> lwp188_all_x9.out

python optim.py -D lwp188_CCO.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_CCO.out
python optim.py -D lwp188_HHb.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_HHb.out
python optim.py -D lwp188_P.optjob scratch/lwp/lwp188_p_fix_pre.csv | tee lwp188_P.out
python optim.py -D lwp188_pH.optjob scratch/lwp/lwp188_p_fix.csv | tee lwp188_pH.out

tail -n 2 lwp188_CCO.out > lwp188_all_x0.out
tail -n 3 lwp188_HHb.out >> lwp188_all_x0.out
tail -n 3 lwp188_P.out >> lwp188_all_x0.out
tail -n 1 lwp188_pH.out >> lwp188_all_x0.out
