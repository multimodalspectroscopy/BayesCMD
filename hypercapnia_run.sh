#!/bin/bash
usage="$(basename "$0") [-h] [-l n] [-b m] -- Run hypercapnia batches of length n, m times

where:
    -h  show this help text
    -l  length of each run
    -b  number of batches/runs to do.
        Should not be more than number of cores."

while getopts ':h:l:b:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    l) RUN_LENGTH=$OPTARG
       ;;
    b) BATCHES=$OPTARG
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

for ((i = 1; i <= $BATCHES; i++));
do
  INPUT_FILE='data/hx01.csv'
  DATE=`date '+%d%m%yT%H:%M:%S'`
  LOG_FILE="data/output-$DATE.txt"
  echo "sysout to $LOG_FILE"
  echo "Running batch $i of length $RUN_LENGTH"
  if [ $i -eq 1 ]; then
    python hypercapnia_batch.py $INPUT_FILE $RUN_LENGTH | tee $LOG_FILE &
    sleep 1m
  else
    python hypercapnia_batch.py $INPUT_FILE $RUN_LENGTH >/dev/null &
    sleep 1m
  fi
done
