import nltk  # Install NLTK by 'conda install -c anaconda nltk' if necessary

nltk.download('punkt') # You need to download an additional package first time.

text = "This is Prof. Choi's lecture. His class doesn't start 9:00 A.M. but there are 1,000 examples,pratice,and homeworks."

# NLTK's recommended word tokenizer, TreebankWordTokenizer + PunktSentenceTokenizer
tokens_word = nltk.tokenize.word_tokenize(text)
print('### NLTK word tokenizer')
print(tokens_word)

# NLTK's recommended sentence tokenizer, PunktSentenceTokenizer
tokens_sent = nltk.tokenize.sent_tokenize(text)
print('### NLTK sentence tokenizer')
print(tokens_sent)
