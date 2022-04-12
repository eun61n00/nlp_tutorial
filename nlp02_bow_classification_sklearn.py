from sklearn import (datasets, feature_extraction, svm, metrics)

# Load the 20 newsgroup dataset
remove = ('headers', 'footers', 'quotes')
train = datasets.fetch_20newsgroups(subset='train', remove=remove)
test  = datasets.fetch_20newsgroups(subset='test',  remove=remove)

# Train the vectorizer
vectorizer = feature_extraction.text.TfidfVectorizer() # Try 'CountVectorizer()' (Accuracy: 0.108/0.093)
vectorizer.fit(train.data)

# Vectorize the training and test data
train_vectors = vectorizer.transform(train.data)
test_vectors = vectorizer.transform(test.data)

# Train the model
model = svm.SVC()
model.fit(train_vectors, train.target)
train_predict = model.predict(train_vectors)
train_accuracy = metrics.balanced_accuracy_score(train.target, train_predict)

# Test the model
test_predict = model.predict(test_vectors)
test_accuracy = metrics.balanced_accuracy_score(test.target, test_predict)

print(f'* Training accuracy: {train_accuracy:.3f}') # Accuracy: 0.969
print(f'* Test accuracy: {test_accuracy:.3f}')      # Accuracy: 0.643
