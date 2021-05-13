import numpy as np
import scipy as sp
from sklearn import datasets
from nlp02_onehot_word import tokenize, build_vocab
from nlp02_onehot_doc  import doc2indices, doc2onehot, inverse_vocab, decode_vector

def doc2bow_hit(doc, vocab):
    indices = doc2indices(doc, vocab)
    doc_vector = sp.sparse.lil_matrix((1, len(vocab)), dtype=np.int64)
    for w in indices:
        doc_vector[0,w] = 1
    return doc_vector

def doc2bow_count(doc, vocab):
    indices = doc2indices(doc, vocab)
    doc_vector = sp.sparse.lil_matrix((1, len(vocab)), dtype=np.int64)
    for w in indices:
        doc_vector[0,w] += 1
    return doc_vector

def build_dcount(docs, vocab):
    dcount = np.zeros(len(vocab), dtype=np.int64)
    for idx, doc in enumerate(docs):
        hit_vector = doc2bow_hit(doc, vocab)
        dcount += hit_vector
    return np.array(dcount)[0] # Make 'np.matrix' to 'np.array'

def build_idf(docs, vocab):
    dcount = build_dcount(docs, vocab)
    n_docs = len(docs)
    idf = np.log((1 + n_docs) / (1 + dcount)) + 1
    return idf

def doc2bow_tfidf(doc, vocab, idf):
    count = doc2bow_count(doc, vocab)
    tf = count.astype(np.float64) / count.sum()
    return tf.multiply(idf).tolil() # TF * IDF


if __name__ == '__main__':
    # Load the 20 newsgroup dataset
    remove = ('headers', 'footers', 'quotes')
    train = datasets.fetch_20newsgroups(subset='train', remove=remove)

    # Build a vocaburary and its document frequency
    vocab = build_vocab(train.data)
    vocab_inv = inverse_vocab(vocab)
    idf = build_idf(train.data, vocab)

    # Test various document vectorizations
    doc_onehot = doc2onehot(train.data[0], vocab)
    doc_vectors = [
        {'name': 'Hit',    'vector': doc2bow_hit(train.data[0], vocab)},
        {'name': 'Count',  'vector': doc2bow_count(train.data[0], vocab)},
        {'name': 'TF-IDF', 'vector': doc2bow_tfidf(train.data[0], vocab, idf)},
    ]
    doc_decode = decode_vector(doc_vectors[-1]['vector'], vocab)

    print('### The test document')
    print('* The number of words: ', len(tokenize(train.data[0])))
    print(train.data[0])

    print('### The document vectors')
    print(f'* One-hot: {doc_onehot.shape} with {doc_onehot.count_nonzero()} non-zeros')
    for dv in doc_vectors:
        print(f'* {dv["name"]}: {dv["vector"].shape} with {dv["vector"].count_nonzero()} non-zeros')
    head = f'| {"WORD":12} | '                        # Print the header of words
    sept = '| ' + '-' * 12 + ' | '                    # Print the saperator of words
    for dv in doc_vectors:
        head += f'{dv["name"]:6} | '                  # Print headers of elements
        sept += '-' * 6 + ' | '                       # Print separators of elements
    print(head)
    print(sept)
    _, cols = doc_vectors[-1]['vector'].nonzero()     # Ignore the row indices of non-zero elements (all 0)
    for idx in cols:
        text = f'| {vocab_inv[idx]:12} | '            # Print the idx-th word
        for dv in doc_vectors:
            text += f"{dv['vector'][0,idx]:>6.3f} | " # Print the idx-th element
        print(text)

    print('### The decoded document by bag-of-words')
    print('* The number of words: ', len(doc_decode))
    print(doc_decode)
