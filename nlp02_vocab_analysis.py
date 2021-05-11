import numpy as np
from sklearn import datasets
from nlp02_onehot_word import build_vocab
from nlp02_onehot_doc  import inverse_vocab
from nlp02_bow_doc     import build_docfreq

# Load the 20 newsgroup dataset
remove = ('headers', 'footers', 'quotes')
train = datasets.fetch_20newsgroups(subset='train', remove=remove)

# Build a vocaburary and its document frequency
vocab = build_vocab(train.data)
vocab_inv = inverse_vocab(vocab)
df = build_docfreq(train.data, vocab)

# Print statistics of the vocabulary
print('### Statistics of the vocabulary')
print(f'* The number of documents: {len(train.data)}')
print(f'* The size of vocabulary: {len(vocab)}')
print(f'* The averaged number of new words per a document: {len(vocab) / len(train.data):.3f}')
print(f'* The range of document frequency: ({df.min()}, {df.max()})')
print(f'* The average of document frequency: {df.mean():.3f}')

# Print low DF (rare) words
print('### Low DF (rare) words')
for df_value in range(1, 10): # Try to check more words
    indices = np.where(df == df_value)[0]
    print(f'* DF = {df_value}')
    print(f'  * {len(indices)} words ({len(indices) / len(vocab) * 100:.1f} % of vocabulary)')
    words = [vocab_inv[idx] for idx in indices[np.random.randint(0, len(indices), 5)]]
    print(f'  * 5 random examples: {words}')
# The rare words are dominant (DF 1: 67.8%, DF 2: 10.5%, DF 3: 4.5%).
# They are typos (e.g. aswer), codes (e.g. a</9&zy<hgd), and numbers (e.g. 02:30:21)

# Print high DF (common) words
print('### High DF (common) words')
df_topk = np.sort(df)[::-1][:100] # Try to check more words
for df_value in np.unique(df_topk)[::-1]:
    indices = np.where(df == df_value)[0]
    print(f'* DF = {df_value}')
    print(f'  * {df_value / len(train.data) * 100:.1f} % documents contain the following words.')
    words = [vocab_inv[idx] for idx in indices]
    print(f'  * {len(words)} words: {words}')
# The common words are mostly stopwords (e.g. the, to, and, of, in).
