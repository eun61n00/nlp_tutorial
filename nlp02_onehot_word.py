import numpy as np
import scipy as sp
from sklearn import datasets

def tokenize(doc):
    IGNORE = ' .,?!":;=+-*/_<>(){}[]`~@#$%^&\\|' + "'"
    doc = doc.lower().replace('\n', ' ').replace('\t', ' ').replace(',', ' ').replace('.', ' ')
    tokens = [word.strip(IGNORE) for word in doc.split(' ') if len(word) > 0]
    return tokens

def build_vocab(docs, min_len=2, stopwords=None, tokenizer=tokenize):
    words = set()                                         # Build as a set
    for doc in docs:                                      # Combine token sets
        words |= {token for token in tokenizer(doc) if len(token) >= min_len}
    if stopwords is not None:
        words -= set(stopwords)                           # Exclude the stopword set
    vocab = {word: idx for idx, word in enumerate(words)} # Build as a dictonary
    return vocab

def word2onehot(word, vocab):
    onehot = np.zeros(len(vocab))
    if word in vocab:
        onehot[vocab[word]] = 1
    return onehot


if __name__ == '__main__':
    # Load the 20 newsgroup dataset
    remove = ('headers', 'footers', 'quotes')
    train = datasets.fetch_20newsgroups(subset='train', remove=remove)

    # Build a vocaburary
    vocab = build_vocab(train.data)

    # Test the word vectorization
    word_test = 'language'
    word_vector = word2onehot('language', vocab)
    word_sparse = sp.sparse.lil_matrix(word_vector) # cf. Row-based linked list sparse matrix
    print(f"* The test word: '{word_test}' (index: {vocab[word_test]})")
    print(f'* Its dense  one-hot vector: {word_vector} (where: {np.where(word_vector == 1)[0][0]})')
    print(f'* Its sparse one-hot vector: {word_sparse}')
