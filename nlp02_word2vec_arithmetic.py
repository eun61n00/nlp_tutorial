import spacy
from sklearn.metrics.pairwise import cosine_similarity

A, B, C = 'puppy', 'dog', 'cat'
top_k = 5

# Find similar vectors roughly
nlp = spacy.load('en_core_web_lg')
query_vector = nlp.vocab.get_vector(A) - nlp.vocab.get_vector(B) + nlp.vocab.get_vector(C)
similar_words = []
for (word_id, word_vector) in nlp.vocab.vectors.items():
    word = nlp.vocab.strings[word_id]
    if word.islower():
        similarity = cosine_similarity([query_vector], [word_vector])[0,0]
        if similarity > 0.5:                          # Cosine similarity more than 0.5
            if word != A and word != B and word != C: # Exclude the same word
                similar_words.append((word_id, similarity))

# Print top-k similar words
print(f"### Finding words for '{A}' - '{B}' + '{C}'")
if len(similar_words) > 0:
    sorted_words = sorted(similar_words, key=lambda x: x[1], reverse=True)
    for rank in range(min(top_k, len(sorted_words))):
        word = nlp.vocab.strings[sorted_words[rank][0]]
        print(f'{rank+1}. {word} ({sorted_words[rank][1]:.3f})')
else:
    print('* Not exist in the vocabulary')
