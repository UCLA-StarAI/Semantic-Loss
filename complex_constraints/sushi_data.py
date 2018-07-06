import numpy as np
from numpy.random import permutation
# Global constants that tell us how to split the indices into data and labels
DATA_IND = [1, 2, 3, 5, 7, 8]
LABEL_IND = [4, 6, 9, 10]

class SushiData():
    def __init__(self, data_path):
        np.random.seed(0)
        data = []
        labels = []
        with open(data_path) as file:
            for line in file:
                tokens = line.strip().split(',')
                # Doesn't have enough entries, isn't data
                if len(tokens) < 10: continue
                # First digit is useless
                ranking = [int(x) for x in tokens[1:]]
                cur_data = []
                cur_label = []
                for item in ranking:
                    if item in DATA_IND:
                        cur_data.append(item)
                    else:
                        cur_label.append(item)
                data.append(to_perm_matrix(cur_data, DATA_IND))
                labels.append(to_perm_matrix(cur_label, LABEL_IND))

        # We're going to split 60/20/20 train/valid/test
        # Nonrandom version
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


def to_perm_matrix(ranking, items):
    # We're going to flatten along the rows, i.e. entries 0-4 are a row (the one hot ranking of the first item), 5-9, etc.
    ret = []
    n = len(items)
    for item in items:
        ret.extend(to_one_hot(ranking.index(item), n))

    return ret

def to_one_hot(dense, n):
    one_hot = np.zeros(n)
    one_hot[dense] = 1
    return one_hot

def to_pairwise_comp(dense, n):
    """ Takes in an array of flattened permutation matrices (i.e. vector of length n^2), returns array of vectors of length n*(n-1)/2, with first element being if first item ranked above second, then first ranked above third,..., second ranked above third, second ranked above fourth,..., fourth rankedabove fifth
    """
    ret = []
    n = int(np.sqrt(dense.shape[1]))
    for i in xrange(dense.shape[0]):
        ordering = from_perm_matrix(dense[i, :], n)
        pairwise = []
        for first in xrange(1, n):
            for second in xrange(first+1, n+1):
                pairwise.append(int(ordering.index(first) < ordering.index(second)))
        ret.append(pairwise)
    return np.array(ret)

def from_perm_matrix(perm_matrix, n):
    """Takes a flattened perm matrix, returns a ranking (numbers 1-n)"""
    indices = [ind for ind, val in enumerate(perm_matrix) if val == 1]
    return [x/n+1 for x in sorted(indices, key=lambda x: x%n)]


if __name__ == '__main__':
    s = SushiData('sushi.soc')
    print from_perm_matrix(np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]), 5)
    print to_pairwise_comp(np.array([[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]]), 5)
