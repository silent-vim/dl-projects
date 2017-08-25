import logging
from six.moves import urllib
import numpy as np
import os
import zipfile
import collections

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class InputData:
    def __init__(self, vocabulary_size):
        self.vocabulary_size = vocabulary_size
        filename = self.maybe_download('http://mattmahoney.net/dc/', 'text8.zip', 31344016)
        self.words = self.read_data(filename)
        logger.info('Data size: %d' % len(self.words))
        self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset()
        logger.info('Most common words (+UNK) %s' % self.count[:5])
        logger.info('Sample Data %s' % self.data[:10])
        del self.words

    @staticmethod
    def maybe_download(url, filename, expected_bytes):
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.exists(filename):
            filename, _ = urllib.urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            logger.info('Found and verified %s' % filename)
        else:
            logger.info('Stat size:%d' % statinfo.st_size)
            raise Exception(
                'Failed to verify ' + filename + '. Can you get to it with a browser?')
        return filename

    @staticmethod
    def read_data(filename):
        f = zipfile.ZipFile(filename)
        for name in f.namelist():
            return f.read(name).split()
        f.close()

    def build_dataset(self):
        count = [['UNK', -1]]
        count.extend(collections.Counter(self.words).most_common(self.vocabulary_size - 1))
        dictionary = {}
        for word, _ in count:
            dictionary[word] = len(dictionary)

        data = []
        unk_count = 0
        for word in self.words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = {v: k for k, v in dictionary.iteritems()}
        return data, count, dictionary, reverse_dictionary

    def generate_batch_cbow(self, data_index, batch_size, skip_window):
        # context_window is the total number of words around the target
        context_size = 2 * skip_window
        batch = np.ndarray(shape=(batch_size, context_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int64)
        span = 2 * skip_window + 1  # [ context_window target context_window ]
        buffer = collections.deque(maxlen=span)

        for _ in range(span):
            buffer.append(self.data[data_index])
            data_index = (data_index + 1) % len(self.data)

        for i in range(batch_size):
            # context tokens are just all the tokens in buffer except the target
            batch[i, :] = [token for idx, token in enumerate(buffer) if idx != skip_window]
            labels[i, 0] = buffer[skip_window]
            buffer.append(self.data[data_index])
            data_index = (data_index + 1) % len(self.data)
        return batch, labels, data_index
