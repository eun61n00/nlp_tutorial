import scipy as sp
from sklearn import (datasets, svm, metrics)
from nlp02_onehot_word  import build_vocab
from nlp02_bow_doc      import build_idf, doc2bow_hit, doc2bow_count, doc2bow_tfidf

if __name__ == '__main__':
    # Load the 20 newsgroup dataset
    remove = ('headers', 'footers', 'quotes')
    train = datasets.fetch_20newsgroups(subset='train', remove=remove)
    test  = datasets.fetch_20newsgroups(subset='test',  remove=remove)

    # Build a vocaburary and its document frequency
    vocab = build_vocab(train.data)
    idf = build_idf(train.data, vocab)

    # Vectorize training and test data
    dataset_vectors = [
        {'name' : 'Hit',
         'train': sp.sparse.vstack([doc2bow_hit(doc, vocab) for doc in train.data]),
         'test' : sp.sparse.vstack([doc2bow_hit(doc, vocab) for doc in test.data])},
        # Stack document vectors vertically for the whole dataset
        {'name' : 'Count',
         'train': sp.sparse.vstack([doc2bow_count(doc, vocab) for doc in train.data]),
         'test' : sp.sparse.vstack([doc2bow_count(doc, vocab) for doc in test.data])},
        {'name' : 'TF-IDF',
         'train': sp.sparse.vstack([doc2bow_tfidf(doc, vocab, idf) for doc in train.data]),
         'test' : sp.sparse.vstack([doc2bow_tfidf(doc, vocab, idf) for doc in test.data])},
    ]

    # Compare with the SVM classifier
    print('### Classification test (accuracy)')
    for vector in dataset_vectors:
        # Train the model
        model = svm.SVC()
        model.fit(vector['train'], train.target)

        # Test the model
        predict = model.predict(vector['test'])
        accuracy = metrics.balanced_accuracy_score(test.target, predict)
        print(f'* {vector["name"]}: {accuracy:.3f}')