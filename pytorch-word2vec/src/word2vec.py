from collections import deque
from tqdm import tqdm

import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging

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


class CBOW(nn.Module):
    def __init__(self, embedding_size, embedding_dim):
        """

        :param embedding_size: count of nodes which have embedding
        :param embedding_dim: embedding dimension
        """
        super(CBOW, self).__init__()
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_dim
        self.u_embeddings = nn.Embedding(2 * embedding_size - 1, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(2 * embedding_size - 1, embedding_dim, sparse=True)
        self.init_embeddings()

    def init_embeddings(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_u, neg_v):
        losses = []
        embedding_u = []
        for i in range(len(pos_u)):
            embedding_ui = self.u_embeddings(autograd.Variable(torch.LongTensor(pos_u[i])))
            embedding_u.append(np.sum(embedding_ui.data.numpy(), axis=0).tolist())

        embedding_u = autograd.Variable(torch.FloatTensor(embedding_u))
        embedding_v = self.v_embeddings(autograd.Variable(torch.LongTensor(pos_v)))
        score = torch.mul(embedding_u, embedding_v)
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        losses.append(sum(score))

        neg_embedding_u = []
        for i in range(len(neg_u)):
            neg_embedding_ui = self.u_embeddings(autograd.Variable(torch.LongTensor(neg_u[i])))
            neg_embedding_u.append(np.sum(neg_embedding_ui.data.numpy(), axis=0).tolist())

        neg_embedding_u = autograd.Variable(torch.FloatTensor(neg_embedding_u))
        neg_embedding_v = self.v_embeddings(autograd.Variable(torch.LongTensor(neg_v)))
        neg_score = torch.mul(neg_embedding_u, neg_embedding_v)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))

        return -1 * sum(losses)

    def save_embedding(self, ix_to_word, file_name):
        embedding = self.v_embeddings.weight.data.numpy()
        fout = open(file_name + "v", 'w')
        fout.write('%d %d\n' % (len(ix_to_word), self.embedding_dim))
        for wid, w in ix_to_word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


class Word2Vec:
    def __init__(self,
                 input_file_name,
                 output_file_name,
                 embedding_dimension=100,
                 batch_size=100,
                 window_size=5,
                 iteration=5,
                 initial_lr=0.025,
                 min_count=1,
                 using_hs=False,
                 using_neg=False,
                 context_size=2,
                 hidden_size=128,
                 cbow=None,
                 skip_gram=None):
        """

        :param input_file_name: Name of a text data from file, each line is a sentence split with space
        :param output_file_name: Name of the final embedding file
        :param embedding_dimension: Embedding dimensionality
        :param batch_size: The count of word pairs for one forward
        :param window_size: Max skip length between words
        :param iteration: Control multiple training iterations
        :param initial_lr: Initial Learning rate
        :param min_count: The minimal word frequency, words with lower frequency will be filtered
        :param using_hs: Whether to use hierarchical softmax
        :param using_neg:
        :param context_size: Used by the CBOW model
        :param hidden_size:
        :param cbow:
        :param skip_gram:
        """
        logger.info('Loading the input file %s' % input_file_name)
        self.data = InputData(input_file_name, min_count)
        logger.info('Input file loaded.')
        logger.info('Input Data %s' % self.data)
        self.output_file_name = output_file_name
        self.embedding_size = len(self.data.word_to_ix)
        logger.info('Embedding size %d' % self.embedding_size)
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.using_hs = using_hs
        self.using_neg = using_neg
        self.cbow = cbow
        self.skip_gram = skip_gram

        if self.cbow is not None and self.cbow:
            self.cbow_model = CBOW(self.embedding_size, self.embedding_dimension)
            logger.info('CBOW Model %s' % self.cbow_model)
            self.optimizer = optim.SGD(self.cbow_model.parameters(), lr=self.initial_lr)

    def cbow_train(self):
        logger.info('CBOW Training ....')
        pair_count = self.data.evaluate_pair_count(self.context_size * 2 + 1)
        logger.info('Pair count: %d' % pair_count)
        batch_count = self.iteration * pair_count / self.batch_size
        logger.info('Batch count: %d' % batch_count)
        process_bar = tqdm(range(int(batch_count)))
        self.cbow_model.save_embedding(self.data.ix_to_word, 'cbow_embedding.txt')
        for i in process_bar:
            pos_pairs = self.data.get_cbow_batch_all_pairs(self.batch_size, self.context_size)
            pos_pairs, neg_pairs = self.data.get_cbow_pairs_by_neg_sampling(pos_pairs, self.context_size)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [int(pair[1]) for pair in pos_pairs]
            neg_u = [pair[0] for pair in neg_pairs]
            neg_v = [int(pair[1]) for pair in neg_pairs]

            self.optimizer.zero_grad()
            loss = self.cbow_model.forward(pos_u, pos_v, neg_u, neg_v)
            loss.backward()
            self.optimizer.step()
            process_bar.set_description("Loss: %0.8f, lr: %0.6f" % (loss.data[0], self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        logger.info('CBOW trained and saving file ....')
        self.cbow_model.save_embedding(self.data.ix_to_word, self.output_file_name)
        logger.info('CBOW Trained and Saved File')


class InputData:
    """Store data for word2vec, such as word map, huffman tree, sampling table and so on.
      Attributes:
          word_frequency: Count of each word, used for filtering low-frequency words and sampling table
          word_to_ix: Map from word to word id, without low-frequency words.
          ix_to_word: Map from word id to word, without low-frequency words.
          sentence_count: Sentence count in files.
          word_count: Word count in files, without low-frequency words.
    """
    def __init__(self, file_name, min_count):
        self.cbow_count = []
        self.cbow_word_pair_catch = deque()
        self.input_file_name = file_name
        self.input_file = open(self.input_file_name)
        self.sentence_length = 0
        self.sentence_count = 0
        self.word_to_ix = {}
        self.ix_to_word = {}
        self.word_frequency = {}
        self.word_count = 0
        self.sample_table = []
        self.get_words(min_count)
        self.init_sample_table()
        logger.info('Word count: %d' % len(self.word_to_ix))
        logger.info('Sentence Length: %d' % self.sentence_length)

    def get_words(self, min_count):
        word_frequency = {}
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word_to_ix[w] = wid
            self.ix_to_word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word_to_ix)

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = np.array(self.sample_table)

    def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) \
               - (self.sentence_count - 1) * (1 + window_size) * window_size

    def get_cbow_batch_all_pairs(self, batch_size, context_size):
        while len(self.cbow_word_pair_catch) < batch_size:
            for _ in range(10000):
                self.input_file = open(self.input_file_name)
                sentence = self.input_file.readline()
                if sentence is None or sentence == '':
                    continue

                word_ids = []
                for word in  sentence.strip().split(' '):
                    try:
                        word_ids.append(self.word_to_ix[word])
                    except:
                        continue

                for i, u in enumerate(word_ids):
                    contentw = []
                    for j, v in enumerate(word_ids):
                        assert u < self.word_count
                        assert v < self.word_count
                        if i == j:
                            continue
                        elif max(0, i - context_size + 1) <= j <= min(len(word_ids), i + context_size - 1):
                            contentw.append(v)
                    if len(contentw) == 0:
                        continue
                    self.cbow_word_pair_catch.append((contentw, u))

        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.cbow_word_pair_catch.popleft())
        return batch_pairs

    def get_cbow_pairs_by_neg_sampling(self, pos_word_pair, count):
        neg_word_pair = []
        for pair in pos_word_pair:
            neg_v = np.random.choice(self.sample_table, size=count)
            neg_word_pair += zip([pair[0]] * count, neg_v)
        return pos_word_pair, neg_word_pair


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input_file_name", help="Name of the input file",
                                 action="store", dest="input_file_name")
    argument_parser.add_argument("--output_file_name", help="Name of the output file",
                                 action="store", dest="output_file_name")
    args = argument_parser.parse_args()
    word2vec = Word2Vec(input_file_name=args.input_file_name,
                        output_file_name=args.output_file_name,
                        cbow=True,
                        skip_gram=False,
                        context_size=2)
    torch.set_num_threads(8)
    word2vec.cbow_train()
