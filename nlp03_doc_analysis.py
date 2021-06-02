import matplotlib.pyplot as plt
import torchtext
from sklearn import datasets
import collections

# Load the 20 newsgroup dataset
remove = ('headers', 'footers', 'quotes')
train_raw = datasets.fetch_20newsgroups(subset='train', remove=remove)

# Prepare the vocabulary
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
counter = collections.Counter()
for doc in train_raw.data:
    counter.update(tokenizer(doc))
vocab = torchtext.vocab.Vocab(counter, min_freq=5)
doc2index = lambda doc: [vocab[token] for token in tokenizer(doc)]
print('* The size of vocabulary: ', len(vocab))

# Plot the histgram of document length
train_doclen = [len(doc2index(doc)) for doc in train_raw.data]
plt.hist(train_doclen, bins=100, range=(0, 1000))
plt.xlabel('The document length [# of words]')
plt.ylabel('Frequency')
ax = plt.gca().twinx() # Add another Y-axis
ax.tick_params(axis='y', colors='r')
plt.hist(train_doclen, bins=100, range=(0, 1000), color='r', density=True, cumulative=True, histtype='step')
plt.ylabel('Ratio', color='r')
plt.xlim(0, 1000)
plt.ylim(0, 1)
plt.show()