import numpy as np

from keras import utils
from config import PAD_TOKEN


class DataGenerator(utils.Sequence):
    def __init__(self, x, y, max_len, vocab_size, batch_size, wtoi):
        self.x = x
        self.y = y
        self.max_length = max_len
        self.vocab_size = vocab_size
        self.total_data_size = len(x[0])
        self.batch_size = batch_size
        self.wtoi = wtoi
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(self.total_data_size / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indices = np.arange(self.total_data_size)

    def __getitem__(self, index):
        indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        this_batch_size = len(indices)
        decoder_input = self.x[1][indices]
        decoder_target = np.array(
            [
                np.append(x[1:], self.wtoi[PAD_TOKEN])
                for x in self.x[1][indices]
            ]
        )
        batch_x = [self.x[0][indices], decoder_input, decoder_target]
        return batch_x, None
