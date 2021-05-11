import spacy
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_lg')         # Download a larger model via 'python -m spacy download en_core_web_lg'
tokens = nlp('dog cat apple banana asdfg') # Try your examples

# Print word vectorization
print('### Word2vec vectorization')
print('* Dimension: ', len(tokens[0].vector))
print('| WORD   | HAS_VEC | NORM   | IS_OOV |')
print('| ------ | ------- | ------ | ------ |')
for tk in tokens:
    print(f'| {tk.text:6} | {tk.has_vector:7} | {tk.vector_norm:>6.3f} | {tk.is_oov:6} |')

# Print the default similarity of several pairs
print('### The default vs. consine similarity')
for (i, j) in [(0, 1), (0, 2), (0, 4)]:
    default = tokens[i].similarity(tokens[j])
    cosine  = cosine_similarity([tokens[i].vector], [tokens[j].vector])[0,0]
    print(f'* {tokens[i].text} - {tokens[j].text}: Default {default:.3f} vs. cosine {cosine:.3f}') # They are same!

# Plot a confusion matrix of the cosine similarity
word_vectors = [tk.vector for tk in tokens]
word_labels  = [tk.text for tk in tokens]
disp = ConfusionMatrixDisplay(cosine_similarity(word_vectors), display_labels=word_labels)
disp.plot()
plt.title('Consine similarity')

# Plot a confusion matrix of the Euclidean distance
disp = ConfusionMatrixDisplay(euclidean_distances(word_vectors), display_labels=word_labels)
disp.plot()
plt.title('Euclidean distance')
