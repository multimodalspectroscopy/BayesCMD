#!/bin/bash
# just concatenate the optimiser output files
tail -n 2 lwp180_CCO.out > lwp180_all.out
tail -n 3 lwp180_HHb.out >> lwp180_all.out
tail -n 3 lwp180_P.out >> lwp180_all.out
tail -n 1 lwp180_pH.out >> lwp180_all.out
tail -n 2 lwp188_CCO.out > lwp188_all.out
tail -n 3 lwp188_HHb.out >> lwp188_all.out
tail -n 3 lwp188_P.out >> lwp188_all.out
tail -n 1 lwp188_pH.out >> lwp188_all.out
