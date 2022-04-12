import spacy
from sklearn.metrics.pairwise import cosine_similarity

query = 'dog' # Try your examples
top_k = 5

# Find similar vectors roughly
nlp = spacy.load('en_core_web_lg')
query_vector = nlp.vocab.get_vector(query) # Get the word vector directly
similar_words = []
for (word_id, word_vector) in nlp.vocab.vectors.items(): # .vectors: Dict (ID-vector)
    word = nlp.vocab.strings[word_id]                    # .strings: Dict (ID-text)
    if word.islower():
        similarity = cosine_similarity([query_vector], [word_vector])[0,0]
        if similarity > 0.5:  # Cosine similarity more than 0.5
            if word != query: # Exclude the same word
                similar_words.append((word_id, similarity))

# Print top-k similar words
print(f"### Similar words to '{query}'")
if len(similar_words) > 0:
    sorted_words = sorted(similar_words, key=lambda x: x[1], reverse=True)
    for rank in range(min(top_k, len(sorted_words))):
        word = nlp.vocab.strings[sorted_words[rank][0]]  # .strings: Dict (ID-text)
        print(f'{rank+1}. {word} ({sorted_words[rank][1]:.3f})')
else:
    print('* Not exist in the vocabulary')
# Try compare results with 'https://explosion.ai/demos/sense2vec'
#                      and 'https://projector.tensorflow.org/'
