from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py as hf
import sys, os
import argparse

from utils import *
from loopalgo import *

### GLOBAL VARS
L = 32

parser = argparse.ArgumentParser(description='Create Markov Chain transition dataset (Type I)')
parser.add_argument('--num_states', dest='num_init_states', default=10, type=int, help='Number of starting states')
parser.add_argument('--max_loopsize', dest='MAX_LOOP_LENGTH', default=12, type=int, help='Assign maximum loop size')
parser.add_argument('--out', dest='OUTPUT_NAME', default='MarkovSet.h5', type=str, help='Name of output dataset')
FLAGS = parser.parse_args()

# number of states from init
num_init_states = FLAGS.num_init_states
MAX_LOOP_LENGTH = FLAGS.MAX_LOOP_LENGTH
OUTPUT_NAME = FLAGS.OUTPUT_NAME

print ('Create {} initial states with maximum loop size is {}'.format(num_init_states, MAX_LOOP_LENGTH))

ices = get_ices()[:num_init_states]
output = hf.File(OUTPUT_NAME, 'w')

# Whole dataset
for idx in range(num_init_states):
    fs = get_filelist(source_idx=idx, prefix='loopsites')
    print ('Load {} loop files with init state {}'.format(len(fs), idx))
    loop_ops = read_filelist(fs)
    num_loop_ops = len(loop_ops)
    print ('{} individual loops in can act on idx={} ice states'.format(len(loop_ops), idx))

    num_skipped = 0
    # One transition
    S0 = ices[idx].copy()
    St = ices[idx].copy()
    padded_loops = []
    icestates = []
    E0 = cal_energy(S0, L)

    for loop in loop_ops:
        current_loop_length = len(loop)
        padding_length = MAX_LOOP_LENGTH - current_loop_length
        if (padding_length < 0): 
            print ('Skip large loop: {}'.format(current_loop_length))
            continue
        is_accept, new_state = transit(St, loop, L, prefix='loopsites')
        if is_accept:
            St = new_state
        else:
            num_skipped += 1
            continue

        # Save to buffer 
        padded_loop = [site for site in loop] + padding_length * [0]
        padded_loops.append(padded_loop)
        icestates.append(St)

    print ('Done index {} markov chain with {} transitions'.format(idx, num_loop_ops-num_skipped))
    config_diff = int(np.sum(np.abs(St-S0))/2)
    print ('Configuration differences: {} ({})'.format(config_diff, config_diff/1024))
    # check the process maintains in ice condition
    Et = cal_energy(St, L)
    energy_diff = Et-E0
    if (energy_diff == 0.0):
        print ('Keeps in Icestates')
        images_name = '_'.join(['MC',str(idx),'states'])
        sequence_name =  '_'.join(['MC',str(idx),'loops'])
        output[images_name] = icestates
        output[sequence_name] = padded_loops

output.close()
print ('Done. Save dataset to {}'.format(OUTPUT_NAME))