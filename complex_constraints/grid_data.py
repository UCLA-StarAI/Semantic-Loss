import numpy as np
import random
from numpy.random import permutation
class GridData():
    def __init__(self, data_path):
        np.random.seed(0)
        data = []
        labels = []
        with open(data_path) as file:
            for line in file:
                tokens = line.strip().split(',')
                if(tokens[0] != ''):
                    removed = [int(x) for x in tokens[0].split('-')]
                else:
                    removed = []

                inp = [int(x) for x in tokens[1].split('-')]
                paths = tokens[2:]
                data.append(np.concatenate((to_one_hot(removed, 24, True), to_one_hot(inp, 16))))
                pathind = 0
                if len(paths) > 1:
                    pathind = random.randrange(len(paths))
                path = [int(x) for x in paths[0].split('-')]
                labels.append(to_one_hot(path, 24))


        # We're going to split 60/20/20 train/test/validation
        perm = permutation(len(data))
        train_inds = perm[:int(len(data)*0.6)]
        valid_inds = perm[int(len(data)*0.6):int(len(data)*0.8)]
        test_inds = perm[int(len(data)*0.8):]
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.train_data = self.data[train_inds, :]
        self.valid_data = self.data[valid_inds, :]
        self.test_data = self.data[test_inds, :]
        self.train_labels = self.labels[train_inds, :]
        self.valid_labels = self.labels[valid_inds, :]
        self.test_labels = self.labels[test_inds, :]

        # Count what part of the batch we're attempt
        self.batch_ind = len(train_inds)
        self.batch_perm = None

        # print self.data.shape
        # print self.labels.shape
        # print self.valid_data.shape
        # print self.valid_labels.shape
        # print self.train_data.shape
        # print self.train_labels.shape
        # print self.test_data.shape
        # print self.test_labels.shape
        np.random.seed()

    def get_batch(self, size):
        # If we're out:
        if self.batch_ind >= self.train_data.shape[0]:
            # Rerandomize ordering
            self.batch_perm = permutation(self.train_data.shape[0])
            # Reset counter
            self.batch_ind = 0

        # If there's not enough
        if self.train_data.shape[0] - self.batch_ind < size:
            # Get what there is, append whatever else you need
            ret_ind = self.batch_perm[self.batch_ind:]
            d, l = self.train_data[ret_ind, :], self.train_labels[ret_ind, :]
            size -= len(ret_ind)
            self.batch_ind = self.train_data.shape[0]
            nd, nl = self.get_batch(size)
            return np.concatenate(d, nd), np.concatenate(l, nl)

        # Normal case
        ret_ind = self.batch_perm[self.batch_ind: self.batch_ind + size]
        return self.train_data[ret_ind, :], self.train_labels[ret_ind, :]

def to_one_hot(dense, n, inv=False):
    one_hot = np.zeros(n)
    one_hot[dense] = 1
    if inv:
        one_hot = (one_hot + 1) % 2
    return one_hot


if __name__ == '__main__':
    gd = GridData('test.out')
