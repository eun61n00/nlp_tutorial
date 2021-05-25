import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets, feature_extraction
from dnn_iris2 import train, evaluate
import time

# Define hyperparameters
EPOCH_MAX = 2000
EPOCH_LOG = 100
OPTIMIZER_PARAM = { 'lr': 1 }
USE_CUDA = torch.cuda.is_available()
RANDOM_SEED = 777

# A two-layer NN model
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size=100, output_size=20):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # 0. Preparation
    torch.manual_seed(RANDOM_SEED)
    if USE_CUDA:
        torch.cuda.manual_seed_all(RANDOM_SEED)
    dev = torch.device('cuda' if USE_CUDA else 'cpu')

    # 1.1. Load the 20 newsgroup dataset
    remove = ('headers', 'footers', 'quotes')
    train_data = datasets.fetch_20newsgroups(subset='train', remove=remove)
    test_data  = datasets.fetch_20newsgroups(subset='test',  remove=remove)

    # 1.2. Train the vectorizer
    vectorizer = feature_extraction.text.TfidfVectorizer(min_df=5, max_df=0.1, stop_words='english')
    vectorizer.fit(train_data.data)

    # 1.3. Vectorize the training and test data
    train_vectors = vectorizer.transform(train_data.data).tocoo()
    test_vectors  = vectorizer.transform(test_data.data).tocoo()

    # 1.4. Tensorize the training and test data
    train_tensors = torch.sparse_coo_tensor([train_vectors.row, train_vectors.col], train_vectors.data, train_vectors.shape, dtype=torch.float32).to(dev)
    train_targets = torch.LongTensor(train_data.target).to(dev)
    test_tensors  = torch.sparse_coo_tensor([test_vectors.row, test_vectors.col], test_vectors.data, test_vectors.shape, dtype=torch.float32).to(dev)
    test_targets  = torch.LongTensor(test_data.target).to(dev)

    # 2. Instantiate a model, loss function, and optimizer
    model = TwoLayerNN(train_tensors.shape[1]).to(dev)
    loss_func = F.cross_entropy
    optimizer = torch.optim.Adadelta(model.parameters(), **OPTIMIZER_PARAM)

    # 3. Train the model
    loss_list = []
    start = time.time()
    for epoch in range(1, EPOCH_MAX + 1):
        train_loss = train(model, [(train_tensors, train_targets)], loss_func, optimizer)
        valid_loss, valid_accuracy = evaluate(model, [(test_tensors, test_targets)], loss_func)

        loss_list.append([epoch, train_loss, valid_loss, valid_accuracy])
        if epoch % EPOCH_LOG == 0:
            elapse = time.time() - start
            print(f'{epoch:>6} ({elapse:>6.2f} sec), TrLoss={train_loss:.6f}, VaLoss={valid_loss:.6f}, VaAcc={valid_accuracy:.3f}')
    elapse = time.time() - start

    # 4. Visualize the loss curves
    plt.title(f'Training and Validation Losses (time: {elapse:.2f} [sec] @ CUDA: {USE_CUDA})')
    loss_array = np.array(loss_list)
    plt.plot(loss_array[:,0], loss_array[:,1] * 1e4, 'r-', label='Training loss')
    plt.plot(loss_array[:,0], loss_array[:,2] * 1e4, 'g-', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss values [1e-4]')
    plt.xlim(loss_array[0,0], loss_array[-1,0])
    plt.grid()
    plt.legend()
    ax = plt.gca().twinx() # Add another Y-axis
    plt.plot(loss_array[:,0], loss_array[:,3], 'g--', label='Validation accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.show()
