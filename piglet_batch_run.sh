#!/bin/bash
usage="$(basename "$0") [-h] [-l n] [-b m] -- Run hypercapnia batches of length n, m times

where:
    -h  show this help text
    -l  length of each run
    -b  number of batches/runs to do.
        Should not be more than number of cores.
    -c  Configuration file."

while getopts ':h:l:b:c:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    l) RUN_LENGTH=$OPTARG
       ;;
    b) BATCHES=$OPTARG
       ;;
    c) CONFIG=$OPTARG
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

INPUT_FILE='data/SA_clean_cropped.csv'
DATE=`date '+%d%m%yT%H:%M:%S'`
LOG_FILE="data/output-$DATE.txt"

if [ $BATCHES -eq 1 ]; then
    python3 batch.py $RUN_LENGTH $INPUT_FILE $CONFIG | tee $LOG_FILE &
else
    for ((i = 1; i <= $BATCHES; i++));
    do

      echo "sysout to $LOG_FILE"
      echo "Running batch $i of length $RUN_LENGTH"
      if [ $i -eq 1 ]; then
        python3 batch.py $RUN_LENGTH $INPUT_FILE $CONFIG | tee $LOG_FILE &
        sleep 1m
      else
        python3 batch.py $RUN_LENGTH $INPUT_FILE $CONFIG >/dev/null &
        sleep 1m
      fi
    done
fi
