# The original code came from Tae-Hwan Jung's NLP tutorial as follows.
# - Reference) https://github.com/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec-Skipgram(Softmax).py

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dnn_iris2 import train

# Define hyperparameters
EPOCH_MAX = 500
EPOCH_LOG = 100
OPTIMIZER_PARAM = {'lr': 0.1}
USE_CUDA = torch.cuda.is_available()
RANDOM_SEED = 1

# Word2Vec (skip-gram) model
class Word2Vec(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(Word2Vec, self).__init__()
        # W and Wt is not traspose relationship.
        self.W = nn.Linear(voc_size, embedding_size, bias=False)  # Input weight  (voc_size -> embedding_size)
        self.Wt = nn.Linear(embedding_size, voc_size, bias=False) # Output weight (embedding_size -> voc_size)

    def forward(self, X):                    # X           : [batch_size, voc_size]
        hidden_layer = self.W(X)             # hidden_layer: [batch_size, embedding_size]
        output_layer = self.Wt(hidden_layer) # output_layer: [batch_size, voc_size]
        return output_layer



if __name__ == '__main__':
    # 0. Preparation
    torch.manual_seed(RANDOM_SEED)
    if USE_CUDA:
        torch.cuda.manual_seed_all(RANDOM_SEED)
    dev = torch.device('cuda' if USE_CUDA else 'cpu')

    # 1. Preapare the dataset
    #    Note) The following data came from 'fruit and juice (skip-gram)' available at https://ronxin.github.io/wevi/
    texts = ['drink apple juice', 'eat orange apple', 'drink rice juice', 'drink juice milk', 'drink milk rice', 'drink water milk', 'orange juice apple', 'apple juice drink', 'rice milk drink', 'milk drink water', 'water drink juice', 'juice drink water']
    words = list(set(' '.join(texts).split()))
    vocab = {word: idx for idx, word in enumerate(words)}

    x = []
    y = []
    onehot_table = np.eye(len(vocab))
    for sentence in texts:
        tokens = sentence.split() # Assume only 3 tokens.
        x.append(onehot_table[vocab[tokens[1]]])
        y.append(vocab[tokens[0]])
        x.append(onehot_table[vocab[tokens[1]]])
        y.append(vocab[tokens[2]])
    x = torch.tensor(x, dtype=torch.float32, device=dev)
    y = torch.tensor(y, dtype=torch.long, device=dev)

    # 2. Instantiate a model, loss function, and optimizer
    embedding_size = 2
    model = Word2Vec(len(vocab), embedding_size).to(dev)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **OPTIMIZER_PARAM)

    # 3. Train the model
    for epoch in range(1, EPOCH_MAX + 1):
        train_loss = train(model, [[x, y]], loss_func, optimizer)
        if epoch % EPOCH_LOG == 0:
            print(f'{epoch:>6}), TrLoss={train_loss:.6f}')

    # 4. Visualize the Word2Vec table
    W, Wt = model.parameters()
    for i, word in enumerate(words):
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()