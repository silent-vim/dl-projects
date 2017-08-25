from input_data import InputData
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import logging
import coloredlogs

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
coloredlogs.install(level='DEBUG', logger=logger)


class Word2Vec:
    def __init__(self, input_data, batch_size, context_window, embedding_size):
        """
        @type input_data: InputData
        """
        self.input_data = input_data
        self.batch_size = batch_size
        self.context_window = context_window  # Number of words to consider on left and right
        self.context_size = 2 * context_window  # Total words in the context left and right
        self.embedding_size = embedding_size
        self.num_negative_sampled = 64
        self.valid_size = 16
        self.valid_window = 100
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self.init = None

        self.train_inputs = None
        self.train_labels = None
        self.optimizer = None
        self.loss = None
        self.normalized_embeddings = None
        self.similarity = None
        self.graph = None
        self.__build_graph()

    def __build_graph(self):
        logger.info('Building Tensorflow Graph ...')
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.context_size], name='training_input')
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='training_labels')
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32, name='validation_dataset')

            with tf.device('/cpu:0'):
                embeddings = tf.Variable(
                    tf.random_uniform([self.input_data.vocabulary_size, self.embedding_size], -1.0, 1.0),
                    name='embeddings'
                )
                embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)
                # take the mean of the context embeddings
                embed_context = tf.reduce_mean(embed, 1)

                # construct the variables for nce loss
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.input_data.vocabulary_size, self.embedding_size],
                                        stddev=1.0 / self.embedding_size**.5),
                    name='nce_weights'

                )
                nce_biases = tf.Variable(tf.zeros([self.input_data.vocabulary_size]))

            # compute the average nce loss for the batch
            # tf.nce_loss automatically draws a new sample of negative labels each time the loss
            # is evaluated
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weights, nce_biases, self.train_labels, embed_context,
                               self.num_negative_sampled, vocabulary_size),
                name='loss_fn'
            )

            # construct the sgd loss with a learning rate
            self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

            # compute the cosine similarity between minibatch examples and all embeddings
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                self.normalized_embeddings, valid_dataset
            )
            self.similarity = tf.matmul(
                valid_embeddings, self.normalized_embeddings, transpose_b=True, name='similarity'
            )

            # add variable initializer
            self.init = tf.initialize_all_variables()

    def train(self, epochs):
        logger.debug('Begin training with Tensorflow ..')
        with tf.Session(graph=self.graph) as session:
            writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())
            # initialize all variables before use
            self.init.run()
            logger.debug('Initialized all variables ...')
            average_loss = 0
            data_index = 0
            for epoch in tqdm(xrange(epochs)):
                batch_inputs, batch_labels, data_index = \
                    self.input_data.generate_batch_cbow(data_index, self.batch_size, self.context_window)
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

                # perform one update step by evaluating the optimizer
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val

                if epoch % 2000 == 0:
                    if epoch > 0:
                        average_loss /= 2000
                        logger.info('Average loss at epoch %s : %s' % (epoch, average_loss))
                        average_loss = 0

                if epoch % 10000 == 0:
                    sim = self.similarity.eval()
                    for i in xrange(self.valid_size):
                        valid_word = self.input_data.reverse_dictionary[self.valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                        log_str = 'Nearest to %s: ' % valid_word
                        for k in xrange(top_k):
                            close_word = self.input_data.reverse_dictionary[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)
            final_embeddings = self.normalized_embeddings.eval()
            writer.close()


if __name__ == '__main__':
    vocabulary_size = 50000
    input_data = InputData(vocabulary_size=vocabulary_size)
    word2vec = Word2Vec(input_data=input_data, batch_size=128, context_window=1, embedding_size=128)
    word2vec.train(epochs=100000)
