#!/bin/bash
# basic script to run sensitivities with jittered ranges for Tracy's paper
python dsim.py lwp180_morris.dsimjob scratch/lwp/lwp180_p_fix.csv | tee sa180.out
python dsim.py lwp188_morris.dsimjob scratch/lwp/lwp188_p_fix.csv | tee sa188.out
