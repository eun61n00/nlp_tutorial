import numpy as np
import scipy as sp
from sklearn import datasets
from nlp02_onehot_word import tokenize, build_vocab

def doc2indices(doc, vocab):
    word_indices = []
    for token in tokenize(doc):
        if token in vocab:
            word_indices.append(vocab[token])
    return word_indices

def doc2onehot(doc, vocab):
    indices = doc2indices(doc, vocab)
    doc_vector = sp.sparse.lil_matrix((len(indices), len(vocab)), dtype=np.int64)
    for s, w in enumerate(indices): # 's'equence index, 'w'ord index
        doc_vector[s, w] = 1
    return doc_vector

def inverse_vocab(vocab):
    return {idx: word for (word, idx) in vocab.items()}

def decode_vector(vector, vocab):
    vocab_inv = inverse_vocab(vocab)
    rs, cs = vector.nonzero() # Get rows and cols of nonzero elements
    words = []
    for i in range(len(rs)):
        words.append(vocab_inv[cs[i]])
    return words


if __name__ == '__main__':
    # Load the 20 newsgroup dataset
    remove = ('headers', 'footers', 'quotes')
    train = datasets.fetch_20newsgroups(subset='train', remove=remove)

    # Build a vocaburary
    vocab = build_vocab(train.data)

    # Test the document vectorization
    doc_vector = doc2onehot(train.data[0], vocab)
    doc_decode = decode_vector(doc_vector, vocab)
    print('### The test document')
    print('* The number of words: ', len(tokenize(train.data[0])))
    print(train.data[0])
    print('### The document vector')
    print('* Shape: ', doc_vector.shape)
    print('* Non-zero elements:')
    print(doc_vector)
    print('### The decoded document')
    print('* The number of words: ', len(doc_decode))
    print(doc_decode)
