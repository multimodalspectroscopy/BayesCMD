#!/bin/bash
# A file to create optimisation logs and then extract the param values and add to a JSON
# Written by: Joshua Russell-Buckland 
# Last updated on: 1/8/2017


usage="$(basename "$0") [-h] [-n MAX_N] -- program to run optimisation code multiple times and compile results into a JSON

where:
    -h  show this help text
    -n  set the max number of runs (Default: 10)"

MAX_N=10
while getopts ':hn:' option; do
  case "$option" in
    h) echo "$usage"
       exit
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


printf "{\n" > ../data/all_opt.json

echo "Running $MAX_N times"
for N in $(seq 0 $MAX_N)
do
  echo "Processing run $N"
  python -u optim.py optjob_files/hx01_BS.optjob ../data/hx01.csv > >(tee -a ../data/optimisation_$N.log)
  printf "\"$N\":" >> ../data/all_opt.json
  python optimisation_compile.py ../data/optimisation_$N.log >>  ../data/all_opt.json
done
truncate -s-2 ../data/all_opt.json
printf "}" >> ../data/all_opt.json

more ../data/all_opt.json
