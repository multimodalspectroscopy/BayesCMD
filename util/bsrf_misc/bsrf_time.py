#! /usr/bin/env python
# hack script to run all BSRF models several times with ech inputs and see how long it takes

# output is redirected to dev/null, because we don't care about the results

import os, os.path, sys
import subprocess
import time

IN = 'examples/bsrf/inputs'
BUILD = 'build'

REPS = 100

MODELS = [ #'BS', 'B1M1', 'B1M2', 'B2M1', 'B2M2',
           #'BSM0', 'BSM1', 'BSM2', 'BSM3',
           #'BSB1', 'BSB2', 'BSB3', 'BSB4',
           #'BrainSignals'
           'BSx', 'B1M1x', 'B1M2x', 'B2M1x', 'B2M2x', 'BrainSignalsx'
         ]

INPUTS = [ 'bsrf_hx01', 'bsrf_hx04', 'bsrf_hx05', 'bsrf_hx07', 'bsrf_hx08',
           'bsrf_hx09', 'bsrf_hx11', 'bsrf_hx12', 'bsrf_hx13', 'bsrf_hx14',
           'autoreg_P_a', 'autoreg_SaO2', 'autoreg_Pa_CO2', 'autoreg_u',
         ]

# NB: we assume models are already built
def time_test ( models ):
    results = {}
    for model in models:
        print >> sys.stderr, 'testing %s' % model
        
        results[model] = {}
        
        for input in INPUTS:
            print >> sys.stderr, 'testing %s with input %s for %d reps' % (model, input, REPS)
            args = [os.path.join(BUILD, model + '.model')]
            args.extend(['-i', os.path.join(IN, input + '.input')])
        
            stdout = os.devnull
            stderr = os.devnull

            start = time.time()
            for rep in range(REPS):        
                with open(stderr, 'w') as err, open(stdout, 'w') as out:
                    ignore = subprocess.call(args, stderr=err, stdout=out)
            end = time.time()
            
            results[model][input] = {'start':start, 'end':end, 'elapsed':end-start, 'mean':(end-start)/float(REPS) }
    
    return results

def write_results(results):
    print 'Model\tInput\tStart\tEnd\tElapsed\tMean'
    for model in results:
        for input in results[model]:
            x = results[model][input]
            print '%s\t%s\t%g\t%g\t%g\t%g' % ( model, input, x['start'], x['end'], x['elapsed'], x['mean'] )

if __name__ == '__main__':
    write_results(time_test(MODELS))

