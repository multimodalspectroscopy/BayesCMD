from run_model import RunModel
from datetime import datetime
import os
import sys
import distutils
import argparse
from bayescmd.util import findBaseDir
BASEDIR = findBaseDir('BayesCMD')

data_path = "/home/buck06191/Dropbox/phd/hypothermia/Piglet Data/filtered_data"

data_files = ['LWP475_filtered.csv', 'LWP479_filtered.csv',
              'LWP481_filtered.csv', 'LWP484_filtered.csv']
