import torch, torchtext
from sklearn.metrics.pairwise import cosine_similarity

word2vec = torchtext.vocab.FastText('ko')     # It downloads a lookup table for word embedding.
words = ['개', '고양이', '사과', '바나나', 'ㅋㅋㅋ'] # Try your examples
word_vectors =[word2vec[word].numpy() for word in words]

# Print the default similarity of several pairs
print('### Consine similarity')
for (i, j) in [(0, 1), (0, 2), (0, 4)]:
    cosine = cosine_similarity([word_vectors[i]], [word_vectors[j]])[0,0]
    print(f'* {words[i]} - {words[j]}: {cosine:.3f}')
