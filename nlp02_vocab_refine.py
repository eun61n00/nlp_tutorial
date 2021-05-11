from sklearn import datasets
from nlp02_onehot_word import build_vocab, tokenize
from nlp02_bow_doc     import build_docfreq

def build_vocab_df(docs, min_len=2, stopwords=None, min_df=2, max_df_ratio=0.5, tokenizer=tokenize):
    words = set()
    for doc in docs:
        words |= {token for token in tokenizer(doc) if len(token) >= min_len}
    if stopwords is not None:
        words -= set(stopwords)                               # Exclude stopwords
    vocab = {word: idx for idx, word in enumerate(words)}     # Build the initial vocabulary

    df = build_docfreq(docs, vocab)                           # Calculate the DF
    max_df = max_df_ratio * len(docs)
    selected = []
    for idx, (word, _) in enumerate(vocab.items()):
        if df[idx] >= min_df and df[idx] <= max_df:           # Check two DF conditions
            selected.append(word)
    vocab = {word: idx for idx, word in enumerate(selected)}  # Re-build the vocabulary
    return vocab

import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def tokenize_spacy(doc):
    parsed = nlp(doc)
    tokens = [word.lemma_ for word in parsed if (not word.is_stop) and (word.pos_ != 'PUNCT' and word.pos_ != 'SPACE')]
    return tokens

def check_vocab(test_vocab, test_words = ['My', 'my', 'is', "don't", 'an', 'the', 'of', 'aswer', 'study', 'studies', 'studied', 'studying']):
    head = f'| {"WORD":10} | '
    sept = '| ' + '-' * 10 + ' | '
    for idx, vocab in enumerate(test_vocab):
        head += f'{idx+1} | '
        sept += '- | '
    print(head)
    print(sept)
    for word in test_words:
        word_exist = f'| {word:10} | '
        for vocab in test_vocab:
            word_exist += ('O' if word in vocab else 'X') + ' | '
        print(word_exist)


if __name__ == '__main__':
    # Load the 20 newsgroup dataset
    remove = ('headers', 'footers', 'quotes')
    train = datasets.fetch_20newsgroups(subset='train', remove=remove)

    # Build vocaburaries
    vocab1 = build_vocab(train.data)
    vocab2 = build_vocab_df(train.data, min_df=5, max_df_ratio=0.1)
    vocab3 = build_vocab_df(train.data, min_df=5, max_df_ratio=0.1, tokenizer=tokenize_spacy)

    # Test the vocaburaries
    vocabs = [vocab1, vocab2, vocab3]
    for idx, vocab in enumerate(vocabs):
        print(f'* The size of vocab{idx+1}: {len(vocab)}')
    check_vocab(vocabs)
