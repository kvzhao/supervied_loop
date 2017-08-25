from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys, os

def get_ices():
    import h5py as hf
    iceset = hf.File('squareice_states_5000x1024.h5', 'r')
    ices = iceset['icestates']
    return ices

def get_filelist(source_idx, prefix='loopstate', dirname='loops'):
    files = os.listdir(dirname)
    filelist = []
    for f in files:
        if str.startswith(f, prefix):
            fname = f.rstrip('.npy')
            trans = fname.split('_')[-1]
            from_idx, to_idx = trans.split('-')
            if (int(from_idx) == source_idx):
                print ('read file: {}'.format(fname))
                filelist.append(f)
    return filelist

def read_filelist(filelist, dirname='loops'):
    loops = []
    for f in filelist:
        loops.extend(np.load('/'.join([dirname, f])))
    return loops

def get_loopsize(loops):
    return [len(np.nonzero(l)[0]) for l in loops]

def combine_isolated_loops(loops):
    try:
        print('number of total loops: {}'.format(len(loops)))
        filtered_loops = []
        marked = {}
        for l in loops:
            checked = True
            for p in np.nonzero(l)[0]:
                if marked.get(p):
                    checked = False
                    break
                else:
                    marked[p] = 1
            if checked:
                filtered_loops.append(l)
                
        print('number of remain loops: {}'.format(len(filtered_loops)))
        return combine_loops(filtered_loops)
    except:
        print ('list is empty')
        return

def combine_loops(loops):
    if loops:
        combined = np.zeros_like(loops[0])
        for l in loops:
            combined += l
            # should prevent combining same loop twice
            # conflict check
        return combined
    else:
        print ('list is empty')
        return

def convert_onehot(loops):
    loops[loops > 0] = 1
    loops[loops < 0] = -1