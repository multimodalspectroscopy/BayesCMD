#!/bin/bash -l
#$ -l h_rt=0:10:00
#$ -N env_test
#$ -wd /home/ucbpjru/Scratch

module load python3/recommended
cd $TMPDIR
export BASEDIR="$HOME/BayesCMD"

python3 $BASEDIR/run_model.py $BASEDIR/data/hx01.csv $BASEDIR/examples/configuration_files/hx01_conf.json --workdir $TMPDIR

tar zcvf $HOME/Scratch/files_from_job_$JOB_ID.tar.gz $TMPDIR
