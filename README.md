## Natural Language Processing Tutorial with Python
_Natural language processing tutorial with Python_ has been initiated to teach undergraduate students in [SeoulTech](https://en.seoultech.ac.kr/) about natural language processing.

This tutorial is an _additional_ part of my four sequels of tutorials. Many examples and practices are connected each other. Please refer to my previous tutorials also.
1. _Python_: From Beginners to Intermediate
2. Programming meets Mathematics
3. Machine Learning Tutorial with _scikit-learn_
4. [Deep Learning Tutorial with _PyTorch_](https://github.com/mint-lab/dl_tutorial)


### Code Examples
#### 1) Text Preprocessing
* **Tokenization**
  * [Simple Word Tokenization by My Hands](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp01_tokenization_simple.py)
  * [Word Tokenization with NLTK](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp01_tokenization_nltk.py)
  * [Word Tokenization with TorchText (spaCy)](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp01_tokenization_torchtext.py)
  * [Morpheme Tokenization with KoNLPy](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp01_tokenization_konlpy.py)
* **Full Preprocessing** (including _tokenization_, _lemmatization_, _POS tagging_, and _stopword filtering_)
  * [Preprocessing with NLTK](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp01_preprocess_nltk.py)
  * [Preprocessing with spaCy](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp01_preprocess_spacy.py)
  * [POS Tagging with KoNLPy](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp01_preprocess_konlpy.py)

#### 2) Text Vectorization: Bag-of-Words
* **One-hot Encoding**
  * [One-hot Vectorization (Word Representation)](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp02_onehot_word.py)
  * [One-hot Vectorization (Document Representation)](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp02_onehot_doc.py)
* **Bag-of-words (Document Representation)**
  * [Visualizing the Inverse Document Frequency](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp02_bow_df.py)
  * [Bag-of-words Vectorization by My Hands](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp02_bow_hand.py)
  * [Text Classification with My Bag-of-words](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp02_bow_classification_hand.py)
  * [Text Classification with scikit-learn](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp02_bow_classification_sklearn.py)
* **Vocabulary Issues**
  * [Vocabulary Analysis using Document Frequency](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp02_vocab_analysis.py)
  * [Vocabulary Refinement using Document Frequency](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp02_vocab_refine.py)
  * [Vocabulary Benchmark for Text Classification](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp02_vocab_bench.py)

#### 3) Text Vectorization: Word Embedding
* **Word Embedding (Word Representation)**
  * [Word2Vec (skip-gram) by My Hands](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp03_word2vec_hand.py)
  * [Word2Vec with spaCy](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp03_word2vec_spacy.py)
  * [Word2Vec with spaCy (Direct Access to the Lookup Table)](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp03_word2vec_spacy_direct.py)
  * [Word2Vec with PyTorch](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp03_word2vec_pytorch.py) and [Its Korean version](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp03_word2vec_pytorch_ko.py)
* **Applications**
  * [Finding Most Similar Words](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp03_app_similar_word.py)
  * [Word Arithmetic](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp03_app_word_arithmetic.py)

#### 4) Text Classification
* **[The 20 Newsgroup Dataset](http://qwone.com/~jason/20Newsgroups/)**
  * [ML methods with Bag-of-words](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp04_bow_sklearn.py) (scikit-learn)
  * [MLP with Bag-of-words](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp04_bow_pytorch.py) (scikit-learn + PyTorch)
  * [MLP with Bag-of-words](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp04_bow_pytorch_only.py) (PyTorch only)
  * [Document Length Analysis](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp04_doc_analysis.py)
  * [RNN with word embedding](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp04_word2vec_rnn.py) (PyTorch)
  * [RNN with _pretrained_ word embedding](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp04_word2vec_rnn_pretrain.py) (PyTorch)
  * [CNN with _pretrained_ word embedding](https://github.com/mint-lab/nlp_tutorial/blob/master/nlp04_word2vec_cnn.py) (PyTorch)

### Author
* [Sunglok Choi](http://mint-lab.github.io/) (sunglok AT seoultech DOT ac DOT kr)
