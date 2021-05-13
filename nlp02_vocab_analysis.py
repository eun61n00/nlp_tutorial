import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from nlp02_onehot_word import build_vocab
from nlp02_onehot_doc  import inverse_vocab
from nlp02_bow_hand    import build_dcount

# Load the 20 newsgroup dataset
remove = ('headers', 'footers', 'quotes')
train = datasets.fetch_20newsgroups(subset='train', remove=remove)

# Build a vocaburary and its document count
vocab = build_vocab(train.data)
vocab_inv = inverse_vocab(vocab)
dcount = build_dcount(train.data, vocab)

# Print statistics of the vocabulary
print('### Statistics of the vocabulary')
print(f'* The number of documents: {len(train.data)}')
print(f'* The size of vocabulary: {len(vocab)}')
print(f'* The averaged number of new words per a document: {len(vocab) / len(train.data):.3f}')
print(f'* The range of document counts: ({dcount.min()}, {dcount.max()})')
print(f'* The average of document counts: {dcount.mean():.3f}')

# Plot the histogram of rare (low document count) words
fig = plt.figure()
dcount10 = dcount[dcount < 10]
plt.hist(dcount10, bins=9, range=(1, 10), align='left')
plt.ylim(0, len(vocab))
plt.xticks(range(1, 10))
plt.xlabel('Document counts')
plt.ylabel('The number of words')
plt.grid()
ax = plt.gca().twinx() # Add another Y-axis
ax.set_ylabel('Ratio [%]')
ax.set_ylim(0, 100)

# Print rare (low document count) words
print('### Rare (low document count) words')
n_example = 20
for dc_value in range(1, 10): # Try to check more words (now dcount < 10)
    indices = np.where(dcount == dc_value)[0]
    print(f'* Document count = {dc_value}')
    print(f'  * {len(indices)} words ({len(indices) / len(vocab) * 100:.1f} % of vocabulary)')
    rand_indx = np.random.randint(0, len(indices), n_example)
    rand_word = [vocab_inv[idx] for idx in indices[rand_indx]]
    print(f'  * Random examples: {rand_word}')
# The rare words are dominant (DC 1: 67.8%, DC 2: 10.5%, DC 3: 4.5%).
# They are typos (e.g. aswer), codes (e.g. a</9&zy<hgd), and numbers (e.g. 02:30:21)

# Plot common words
topk, step = 1000, 50
dcount_pair = [(dc_value, idx) for idx, dc_value in enumerate(dcount)]
dcount_sort = sorted(dcount_pair, key=lambda x: x[0], reverse=True)
fig = plt.figure()
dcount_topk = [dc_value for dc_value, _ in dcount_sort[:topk]]
plt.plot(dcount_topk)
plt.ylim(0, len(train.data))
xticks_vals = range(0, topk+1, step)
xticks_word = [vocab_inv[dcount_sort[x][1]] for x in xticks_vals]
plt.xticks(xticks_vals, xticks_word, rotation=90)
plt.ylabel('Document counts')
plt.grid()
ax = plt.gca().twinx() # Add another Y-axis
ax.set_ylabel('Document frequency')

# Print common (high document count) words
print('### Common (high document count) words')
topk = 100
for rank, (dc_value, word_idx) in enumerate(dcount_sort[:topk]):
    print(f"{rank+1}. '{vocab_inv[word_idx]}' (dcount = {dc_value}, dfreq = {dc_value / len(train.data):.3f})")
# The common words are mostly stopwords (e.g. the, to, and, of, in).

