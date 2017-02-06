#! /usr/bin/env python
# hack script to rebuild all BSRF models and regenerate corresponding hypercapnia outputs
# must be executed in the BCMD home directory

import os, os.path, sys
import subprocess
import tempfile

import numpy
import batch.distance
import batch.inputs

IN = 'examples/bsrf/inputs'
OUT = 'data/bsrf'
BUILD = 'build'
PARSER = 'bparser/bcmd.py'
INFO = 'bparser/info.py'
SUMMARY = 'summary.txt'

BUILDS = [ { 'dir':'examples', 'model':'BrainSignals' },
           { 'dir':'examples/bsrf', 'model':'BS' },
           { 'dir':'examples/bsrf', 'model':'B1M1' },
           { 'dir':'examples/bsrf', 'model':'B1M2' },
           { 'dir':'examples/bsrf', 'model':'B2M1' },
           { 'dir':'examples/bsrf', 'model':'B2M2' },
           { 'dir':'examples/bsrf', 'model':'BSM0' },
           { 'dir':'examples/bsrf', 'model':'BSM1' },
           { 'dir':'examples/bsrf', 'model':'BSM2' },
           { 'dir':'examples/bsrf', 'model':'BSM3' },
           { 'dir':'examples/bsrf', 'model':'BSB1' },
           { 'dir':'examples/bsrf', 'model':'BSB2' },
           { 'dir':'examples/bsrf', 'model':'BSB3' },
           { 'dir':'examples/bsrf', 'model':'BSB4' },
         ]

INPUTS = [ 'bsrf_hx01', 'bsrf_hx04', 'bsrf_hx05', 'bsrf_hx07', 'bsrf_hx08',
           'bsrf_hx09', 'bsrf_hx11', 'bsrf_hx12', 'bsrf_hx13', 'bsrf_hx14',
           'autoreg_P_a', 'autoreg_SaO2', 'autoreg_Pa_CO2', 'autoreg_u',
         ]

DIST_FIELDS = [ 'CCO', 'CBF', 'CMRO2' ]

def summarise ( builds ):
    if len(builds) < 2: return
    
    targets = {}
    results = {}
    
    for input in INPUTS:
        targets[input] = {}
        out = batch.inputs.readCSV(os.path.join(OUT, builds[0]['model'], input + '.out'), False)
        for field in DIST_FIELDS:
            if field in out:
                targets[input][field] = out[field]
    
    for build in builds[1:]:
        model = build['model']
        results[model] = {}
        for input in INPUTS:
            results[model][input] = {}
            out = batch.inputs.readCSV(os.path.join(OUT, build['model'], input + '.out'), False)
            for field in DIST_FIELDS:
                if field in out and field in targets[input]:
                    results[model][input][field] = batch.distance.euclidean(numpy.array(targets[input][field]), numpy.array(out[field]))
    
    with open(os.path.join(OUT, SUMMARY), 'w') as f:
        
        print >> f, 'Model\tInput\tField\tDistance'
        
        for model in results:
            for input in results[model]:
                for field in results[model][input]:
                    print >> f, '%s\t%s\t%s\t%f' % (model, input, field, results[model][input][field])


def regen ( builds ):
    for build in builds:
        print >> sys.stderr, 'parsing %s' % build['model']
        args = ['python', PARSER, '-v', '5', '-p', '-t' ]
        args.extend(['-i', build['dir'], '-n', build['model']] )
        args.extend(['-o', build['model'] + '.model'])
        args.extend(['-d', BUILD])
        args.append(os.path.join(build['dir'], build['model'] + '.modeldef'))
        logname = os.path.join(BUILD, build['model'] + '.log')
        with open(logname, 'w') as stderr:
            result = subprocess.call(args, stderr=stderr)
    
        if result:
            print >> sys.stderr, 'parsing failed with exit code %d (see %s for details)' % (result, logname)
        else:
            print >> sys.stderr, 'documenting %s' % build['model']
            args = ['python', INFO, '-m', '-t', '-H', '-l', '-a' ]
            args.extend(['-d', BUILD])
            args.append(os.path.join(BUILD, build['model'] + '.bcmpl'))
            subprocess.call(args)
            
            print >> sys.stderr, 'compiling %s' % build['model']
            args = ['make', os.path.join(BUILD, build['model'] + '.model'), 'DEBUG=0']
            stdout = os.path.join(os.path.join(BUILD, build['model'] + '.stdout'))
            stderr = os.path.join(os.path.join(BUILD, build['model'] + '.stderr'))
            with open(stderr, 'w') as err, open(stdout, 'w') as out:
                result = subprocess.call(args, stderr=err, stdout=out)
        
            if result:
                print >> sys.stderr, 'compilation failed with exit code %d' % result
            else:
                outdir = os.path.join(OUT, build['model'])
                checkout = outdir
                idx = 1
                while os.path.exists(checkout) and not os.path.isdir(checkout):
                    checkout = '%s_%d' % (outdir, idx)
                    idx += 1
                
                outdir = checkout
                print >> sys.stderr, 'output directory is %s' % outdir
                
                if not os.path.exists(outdir):
                    print >> sys.stderr, 'directory does not exist, attempting to create it'
                    os.makedirs(outdir)
                    
                for input in INPUTS:
                    print >> sys.stderr, 'running %s with input %s' % (build['model'], input)
                    args = [os.path.join(BUILD, build['model'] + '.model')]
                    args.extend(['-i', os.path.join(IN, input + '.input')])
                    args.extend(['-o', os.path.join(outdir, input + '.out')])
                
                    stdout = os.path.join(os.path.join(outdir, input + '.stdout'))
                    stderr = os.path.join(os.path.join(outdir, input + '.stderr'))
                
                    with open(stderr, 'w') as err, open(stdout, 'w') as out:
                        result = subprocess.call(args, stderr=err, stdout=out)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-d':
            summarise(BUILDS)
        else:
            builds = [ x for x in BUILDS if x['model'] in sys.argv[1:] ]
            regen(builds)
            summarise(builds)
    else:
        # default to building everything
        regen(BUILDS)
        summarise(BUILDS)
