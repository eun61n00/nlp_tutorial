import nltk

text = "This is Prof. Choi's lecture. His class doesn't start 9:00 A.M. but there are 1,000 examples,pratice,and homeworks."

nltk.download('wordnet')                     # You need to download an additional package first time.
nltk.download('averaged_perceptron_tagger')  # You need to download an additional package first time.
nltk.download('universal_tagset')            # You need to download an additional package first time.
nltk.download('stopwords')

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer    = nltk.stem.PorterStemmer()
stopwords  = nltk.corpus.stopwords.words('english')

tokens  = nltk.tokenize.word_tokenize(text)           # Tokenization
lemmas  = [lemmatizer.lemmatize(tk) for tk in tokens] # Lemmatization
stems   = [stemmer.stem(tk) for tk in tokens]         # Stemming
pos_def = nltk.pos_tag(tokens)                        # POS tagging with the default tagset
pos_uni = nltk.pos_tag(tokens, tagset='universal')    # POS tagging with the universal tagset
is_stop = [tk in stopwords for tk in tokens]          # Stopword rejection

print(f"{'TOKEN':10} | {'LEMMA':10} | {'STEM':10} | {'POS_DEF':7} | {'POS_UNI':7} | {'STOPWORD'}")
print(f"{'-'*10} | {'-'*10} | {'-'*10} | {'-'*7} | {'-'*7} | {'-'*8}")
for i in range(len(tokens)):
    print(f'{tokens[i]:10} | {lemmas[i]:10} | {stems[i]:10} | {pos_def[i][1]:7} | {pos_uni[i][1]:7} | {is_stop[i]}')
