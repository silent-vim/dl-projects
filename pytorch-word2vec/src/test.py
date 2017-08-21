from sklearn.metrics.pairwise import cosine_similarity

import argparse
import numpy as np
import logging

# create a global logger
logger = logging.getLogger('word2vec_test')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def test_embeddings(input_file_name):
    with open(input_file_name, 'r+') as input_file:
        input_file.readline()
        all_embeddings = []
        all_words = []
        word_to_ix = {}
        for i, line in enumerate(input_file):
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            all_embeddings.append(embedding)
            all_words.append(word)
            word_to_ix[word] = i
        all_embeddings = np.array(all_embeddings)
        while 1:
            word = raw_input('Enter the word: ')
            try:
                wid = word_to_ix[word]
            except:
                logging.error('Cannot find the word %s' % word)
                continue
            embedding = all_embeddings[wid: wid + 1]
            d = cosine_similarity(embedding, all_embeddings)[0]
            d = zip(all_words, d)
            d = sorted(d, key=lambda x: x[1], reverse=True)
            for w in d[:10]:
                if len(w[0]) < 2:
                    continue
                logger.info(w)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input_file_name", help="Name of the input file",
                                 action="store", dest="input_file_name")
    args = argument_parser.parse_args()
    test_embeddings(args.input_file_name)