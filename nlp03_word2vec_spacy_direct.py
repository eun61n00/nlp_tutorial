import spacy
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('en_core_web_lg')
words = ['dog', 'cat', 'apple', 'banana', 'asdfg'] # Try your examples
word_vectors =[nlp.vocab.get_vector(word) for word in words]

# Print the default similarity of several pairs
print('### Consine similarity')
for (i, j) in [(0, 1), (0, 2), (0, 4)]:
    cosine = cosine_similarity([word_vectors[i]], [word_vectors[j]])[0,0]
    print(f'* {words[i]} - {words[j]}: {cosine:.3f}')
