from tqdm import tqdm

import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import os
import urllib
import zipfile
import collections

np.random.seed(12345)

# create a global logger
logger = logging.getLogger('word2vec_application')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.info('Word2Vec Implementation with Pytorch')
VOCABULARY_SIZE = 50000


class CBOW(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension):
        """

        :param embedding_size: count of nodes which have embedding
        :param embedding_dim: embedding dimension
        """
        super(CBOW, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension)
        self.linear = nn.Linear(embedding_dimension, vocabulary_size)
        self.init_embeddings()

    def init_embeddings(self):
        initrange = 0.5 / self.embedding_dimension
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        # print inputs.data.shape
        embedding = self.embeddings(inputs)
        avg_embedding = torch.mean(embedding, dim=1)
        out = self.linear(avg_embedding)
        log_probs = F.log_softmax(out)
        return log_probs
        #return torch.max(log_probs, dim=1, keepdim=True)[1]


class Word2Vec:
    def __init__(self):
        logger.info('CBOW Training ....')
        self.batch_size = 128
        self.embedding_dimension = 128
        self.skip_window = 1

        self.input_data = InputData()
        self.cbow = CBOW(vocabulary_size=VOCABULARY_SIZE, embedding_dimension=self.embedding_dimension)

    def train(self):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adagrad(self.cbow.parameters(), lr=0.001)

        epochs = 100001
        data_index = 0
        for epoch in tqdm(range(epochs)):
            batch_data, batch_labels, data_index = self.input_data.generate_batch_cbow(data_index, self.batch_size, self.skip_window)
            x_values = autograd.Variable(batch_data)
            y_labels = autograd.Variable(batch_labels[:,0])
            predicted = self.cbow(x_values)
            loss = loss_function(predicted, y_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print('[%d/%d] Loss: %.3f' % (epoch + 1, epochs, loss.data.mean()))



class InputData:
    def __init__(self):
        self.vocabulary_size = VOCABULARY_SIZE
        filename = self.maybe_download('http://mattmahoney.net/dc/', 'text8.zip', 31344016)
        self.words = self.read_data(filename)
        logger.info('Data size: %d' % len(self.words))
        self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset()
        logger.info('Most common words (+UNK) %s' % self.count[:5])
        logger.info('Sample Data %s' % self.data[:10])
        skip_window = 1
        self.batch, self.labels, data_index = self.generate_batch_cbow(0, 8, skip_window)
        logger.info('Labels: {0}'.format(self.labels.shape))
        for i in range(8):
            print(self.batch[i, 0], self.reverse_dictionary[self.batch[i, 0]],
                  self.batch[i, 1], self.reverse_dictionary[self.batch[i, 1]],
                  '->', self.labels[i, 0], self.reverse_dictionary[self.labels[i, 0]])

        del skip_window  # remove skip_window setting used for testing

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
        return torch.from_numpy(batch), torch.from_numpy(labels), data_index


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input_file_name", help="Name of the input file",
                                 action="store", dest="input_file_name")
    argument_parser.add_argument("--output_file_name", help="Name of the output file",
                                 action="store", dest="output_file_name")
    args = argument_parser.parse_args()
    # torch.set_num_threads(8)
    word2vec = Word2Vec()
    word2vec.train()