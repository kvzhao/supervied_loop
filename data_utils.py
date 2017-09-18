import numpy as np
import h5py as hf
import sys, os

from utils import *
from loopalgo import *

def get_mask(sequences):
    return [0 if s == 0 else 1 for s in sequences]

def get_batch_mask(batch_seq):
    batch_mask = []
    for seq in batch_seq:
        batch_mask.append(seq)
    return np.array(batch_mask)

def read_markovchain_dataset(dataset_path):
    dataset = hf.File(dataset_path, 'r')
    # How to load multiple mc transitions?
    states = dataset['MC_0_states'][:]
    loops = dataset['MC_0_loops'][:]
    dataset.close()

    return states, loops

class DataReader(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        states, loops = read_markovchain_dataset(self.dataset_path)
        masks = get_batch_mask(loops)

        self.images = states
        self.sequences = loops
        self.masks = masks

        # Login dataset spec
        self._num_of_samples = self.images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_samples(self):
        return self._num_of_samples

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_of_samples)
            np.random.shuffle(perm0)
            self.images = self.images[perm0]
            self.sequences = self.sequences[perm0]
            self.masks = self.masks[perm0]
                # Finsh the epoch
        if start + batch_size > self._num_of_samples:
            self._epochs_completed += 1
            rest_num_samples = self._num_of_samples - start
            images_rest_part = self.images[start:self._num_of_samples]
            sequences_rest_part = self.sequences[start:self._num_of_samples]
            masks_rest_part = self.masks[start:self._num_of_samples]
            # Shuffle
            if shuffle:
                perm = np.arange(self._num_of_samples)
                np.random.shuffle(perm)
                self.images = self.images[perm]
                self.sequences = self.sequences[perm]
                self.masks = self.masks[perm]
                # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_samples
            end = self._index_in_epoch
            images_new_part = self.images[start:end]
            sequences_new_part = self.sequences[start:end]
            masks_new_part = self.masks[start:end]

            # Return input & target sequences (an expansive method)
            sequences = np.concatenate((sequences_rest_part, sequences_new_part), axis=0)
            input_sequences =[]
            target_sequences =[]
            for seq in sequences:
                input_sequences.append([s for s in seq[:-1]])
                target_sequences.append([s for s in seq[1:]])

            return np.concatenate((images_rest_part, images_new_part), axis=0), \
                input_sequences, target_sequences, \
                np.concatenate((masks_rest_part, masks_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            input_sequences =[]
            target_sequences =[]
            for seq in self.sequences[start:end]:
                input_sequences.append([s for s in seq[:-1]])
                target_sequences.append([s for s in seq[1:]])
            return self.images[start:end], input_sequences, target_sequences, self.masks[start:end]