#!/bin/bash
# A file to create optimisation logs and then extract the param values and add to a JSON
# Written by: Joshua Russell-Buckland
# Last updated on: 1/8/2017


usage="$(basename "$0") [-h] [-N number of parameters][-n MAX_N] -- program to run optimisation code multiple times and compile results into a JSON

where:
    -h  show this help text
    -N  Number of parameters being optimised
    -n  set the max number of runs (Default: 10)"

MAX_N=10
pflag=false
while getopts ':hN:n:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    N) pflag=true; Nparams=$OPTARG
       ;;
    n) MAX_N=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

if ! $pflag
then
  echo "Number of parameters not provided."
  exit 1
fi

DATE=`date '+%d%m%yT%H:%M:%S'`
mkdir "../data/$DATE"
echo "Created directory: ../data/$DATE"
printf "{\n" > ../data/$DATE/all_opt.json

echo "Running $MAX_N times"
for N in $(seq 0 $MAX_N)
do
  echo "Processing run $N"
  python -u optim.py optjob_files/hx01_BS.optjob ../data/hx01.csv > >(tee -a ../data/$DATE/optimisation_$N.log)
  printf "\"$N\":" >> ../data/all_opt.json
  python optimisation_compile.py ../data/optimisation_$N.log $Nparams >>  ../data/$DATE/all_opt.json
done
truncate -s-2 ../data/$DATE/all_opt.json
printf "}" >> ../data/$DATE/all_opt.json

more ../data/$DATE/all_opt.json
