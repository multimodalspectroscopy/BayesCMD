export BASEDIR=".."
TMPDIR="./test_data/"

DATAFILE="$BASEDIR/PLOS_paper/data/simulated_smooth_combined_ABP.csv"
CONFIGFILE="$BASEDIR/examples/configuration_files/simulated_parameter_config.json"

start=`date +%s`
python3 $BASEDIR/batch_Bayes/batch.py 500 $DATAFILE $CONFIGFILE --workdir $TMPDIR
echo "Duration: $(($(date +%s)-$start))" > $TMPDIR/timings.txt
