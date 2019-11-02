import keras
import h5py
import random
import numpy as np

class HDF5DataProvider(keras.utils.Sequence):
    def __init__(self, path, idxs, batch_size, shuffle=True):
        self.path = path
        self.idxs = idxs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        """Number of batch in the Sequence."""
        return len(self.idxs)//self.batch_size


    def __getitem__(self, index):
        """Get batch at position x."""
        pos = index*self.batch_size
        batch_idxs = self.idxs[pos:(pos+self.batch_size)]
        batch_idxs = list(zip(batch_idxs, list(range(len(batch_idxs)))))
        # H5PY accepts indices in ascending order only.
        batch_idxs = sorted(batch_idxs, key=lambda x: x[0])
        with h5py.File(self.path, 'r') as f:
            X, y = f['X'], f['y']
            # pos - original position
            batch_idxs, pos = zip(*batch_idxs)
            batch_idxs = list(batch_idxs)
            X, y = X[batch_idxs], y[batch_idxs]
            ss_pos = zip(pos, range(len(pos)))
            sorted_pos = sorted(ss_pos, key=lambda x: x[0])
            _, original_pos = zip(*sorted_pos)
            original_pos = list(original_pos)
            batch_idxs = np.array(batch_idxs)
            return X[original_pos], y[original_pos]


    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.idxs)

    def get_data_shape(self):
        # assumes each rf has the same size
        X, y = self.__getitem__(0)
        return X[0].shape, y[0].shape
