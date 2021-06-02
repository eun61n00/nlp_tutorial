import numpy as np
import matplotlib.pyplot as plt
import torch, torchtext
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets
import collections, time

# Define hyperparameters
EPOCH_MAX = 20000
EPOCH_LOG = 100
OPTIMIZER_PARAM = {'lr': 1}
USE_CUDA = torch.cuda.is_available()
RANDOM_SEED = 777

# A two-layer NN model
class MyTwoLayerNN(nn.Module):
    def __init__(self, vocab_size, embed_size=100, output_size=20):
        super(MyTwoLayerNN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, output_size)

        # Initialize weight variables
        self.embedding.weight.data.uniform_(-0.5, 0.5)
        self.fc.weight.data.uniform_(-0.5, 0.5)
        self.fc.bias.data.zero_()

    def forward(self, x, offset):
        x = self.embedding(x, offset)
        x = F.relu(self.fc(x))
        return x

# Train a model with the given batches with offsets
def train(model, batch_data, loss_func, optimizer):
    model.train()  # Notify layers (e.g. DropOut, BatchNorm) that it’s now training
    train_loss, n_data = 0, 0
    dev = next(model.parameters()).device
    for batch_idx, (x, offset, y) in enumerate(batch_data):
        x, offset, y = x.to(dev), offset.to(dev), y.to(dev)
        optimizer.zero_grad()
        output = model(x, offset)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_data += len(y)
    return train_loss / n_data

# Evaluate a model with the given batches with offsets
def evaluate(model, batch_data, loss_func):
    model.eval()  # Notify layers (e.g. DropOut, BatchNorm) that it’s now testing
    test_loss, n_correct, n_data = 0, 0, 0
    with torch.no_grad():
        dev = next(model.parameters()).device
        for x, offset, y in batch_data:
            x, offset, y = x.to(dev), offset.to(dev), y.to(dev)
            output = model(x, offset)
            loss = loss_func(output, y)
            y_pred = torch.argmax(output, dim=1)

            test_loss += loss.item()
            n_correct += (y == y_pred).sum().item()
            n_data += len(y)
    return test_loss / n_data, n_correct / n_data


if __name__ == '__main__':
    # 0. Preparation
    torch.manual_seed(RANDOM_SEED)
    if USE_CUDA:
        torch.cuda.manual_seed_all(RANDOM_SEED)
    dev = torch.device('cuda' if USE_CUDA else 'cpu')

    # 1.1. Load the 20 newsgroup dataset
    remove = ('headers', 'footers', 'quotes')
    train_raw = datasets.fetch_20newsgroups(subset='train', remove=remove)
    tests_raw = datasets.fetch_20newsgroups(subset='test',  remove=remove)

    # 1.2. Prepare the vocabulary
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english') # Try 'spacy' or your custom tokenizers
    counter = collections.Counter()
    for doc in train_raw.data:
        counter.update(tokenizer(doc))
    vocab = torchtext.vocab.Vocab(counter, min_freq=5)
    doc2index = lambda doc: [vocab[token] for token in tokenizer(doc)]
    print('* The size of vocabulary: ', len(vocab))

    # 1.3. Transform training and test data into indices and their offsets
    def dataset2index(dataset):
        indices = [ ]
        offsets = [0]
        for doc in dataset:
            index = doc2index(doc)
            indices.append(torch.LongTensor(index))
            offsets.append(len(index))
        return torch.cat(indices).to(dev), torch.tensor(offsets[:-1]).cumsum(dim=0).to(dev)

    train_indices, train_offsets = dataset2index(train_raw.data)
    tests_indices, tests_offsets = dataset2index(tests_raw.data)
    train_targets = torch.LongTensor(train_raw.target).to(dev)
    tests_targets = torch.LongTensor(tests_raw.target).to(dev)

    # 2. Instantiate a model, loss function, and optimizer
    model = MyTwoLayerNN(len(vocab)).to(dev)
    loss_func = F.cross_entropy
    optimizer = torch.optim.Adadelta(model.parameters(), **OPTIMIZER_PARAM)
    print('* The number of model parameters: ', sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))

    # 3. Train the model
    loss_list = []
    start = time.time()
    for epoch in range(1, EPOCH_MAX + 1):
        train_loss = train(model, [(train_indices, train_offsets, train_targets)], loss_func, optimizer)
        valid_loss, valid_accuracy = evaluate(model, [(tests_indices, tests_offsets, tests_targets)], loss_func)

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
