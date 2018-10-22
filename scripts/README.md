# Scripts # 

This directory contains a number of useful scripts and examples of how to use various BayesCMD features.

## `legions_scripts/` ##
This includes an example of a job submission script for the legion computing cluster at UCL. For more information, see [the UCL Legion webpage](https://wiki.rc.ucl.ac.uk/wiki/Legion_Quick_Start).

## `results_processing/` ##
This includes an example of how to use the results handling functionality built in to BayesCMD. This will be updated as this module is modified.

## `single_run/` ##
Whilst BayesCMD is intended for Bayesian fitting, which requires many runs, it retains the ability to a run a model a single time. To do so, [`run_model.py`](https://github.com/buck06191/BayesCMD/blob/master/scripts/single_run/run_model.py) can be used, with a BayesCMD config file used to configure the model requirements.

## `steady_state/` ##
BayesCMD can be used to generate steady state simulations using [`run_steadystate.py`](https://github.com/buck06191/BayesCMD/blob/master/scripts/steady_state/run_steadystate.py). This directory also contains examples on how to configure and use this functionality.

## `batch.py` ##
[`batch.py`](https://github.com/buck06191/BayesCMD/blob/master/scripts/batch.py) is the main access point for running the BayesCMD code in multiple batches, as in Legion array jobs. 
