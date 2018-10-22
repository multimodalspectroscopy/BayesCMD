#!/bin/bash -l
#$ -l h_rt=0:20:00
#$ -N just_kaut_healthy
#$ -wd /home/ucbpjru/Scratch
# Set up the job array.  In this instance we have requested 1000 tasks
# numbered 1 to 1000.
#$ -t 1-50

module load python3/recommended
cd $TMPDIR
export BASEDIR="$HOME/BayesCMD"

DATAFILE="$BASEDIR/PLOS_paper/data/simulated_smooth_combined_ABP.csv"
CONFIGFILE="$BASEDIR/examples/configuration_files/just_kaut.json"

start=`date +%s`
python3 $BASEDIR/batch_Bayes/batch.py 1000 $DATAFILE $CONFIGFILE --workdir $TMPDIR
echo "Duration: $(($(date +%s)-$start))" > $TMPDIR/$SGE_TASK_ID.timings.txt

tar -zcvf $HOME/Scratch/batch_$JOB_NAME.$SGE_TASK_ID.tar.gz $TMPDIR
