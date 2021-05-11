from sklearn import (datasets, feature_extraction, svm, metrics)
from nlp02_vocab_refine import tokenize_spacy, check_vocab

# Load the 20 newsgroup dataset
remove = ('headers', 'footers', 'quotes')
train = datasets.fetch_20newsgroups(subset='train', remove=remove)
test  = datasets.fetch_20newsgroups(subset='test',  remove=remove)

# Train vectorizers
vectorizers = [
    {'name': 'Original',        'izer': feature_extraction.text.TfidfVectorizer()},
    {'name': '(5, 0.1)',        'izer': feature_extraction.text.TfidfVectorizer(min_df=5, max_df=0.1)},
    {'name': '(5, 0.1)+StopWd', 'izer': feature_extraction.text.TfidfVectorizer(min_df=5, max_df=0.1, stop_words='english')},
    {'name': '(5, 0.1)+spaCy',  'izer': feature_extraction.text.TfidfVectorizer(min_df=5, max_df=0.1, tokenizer=tokenize_spacy)},
]
for vector in vectorizers:
    vector['izer'].fit(train.data)

# Compare with a simple test
print('### Simple test')
for vector in vectorizers:
    print(f'* {vector["name"]}: {len(vector["izer"].vocabulary_)} words')
check_vocab([vector["izer"].vocabulary_ for vector in vectorizers])

# Compare with the SVM classifier
print('### Classification test (accuracy)')
for vector in vectorizers:
    # Vectorize the training and test data
    train_vectors = vector['izer'].transform(train.data)
    test_vectors  = vector['izer'].transform(test.data)

    # Train the model
    model = svm.SVC()
    model.fit(train_vectors, train.target)

    # Test the model
    predict = model.predict(test_vectors)
    accuracy = metrics.balanced_accuracy_score(test.target, predict)
    print(f'* {vector["name"]}: {accuracy:.3f}')
