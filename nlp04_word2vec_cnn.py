import numpy as np
import matplotlib.pyplot as plt
import torch, torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn import datasets
from dnn_iris2 import train, evaluate
import time

# Define hyperparameters
EPOCH_MAX = 500
EPOCH_LOG = 100
OPTIMIZER_PARAM = {'lr': 1}
DATA_LOADER_PARAM = {'batch_size': 1000, 'shuffle': True}
DATA_MAX_LEN = 200
USE_CUDA = torch.cuda.is_available()
RANDOM_SEED = 777

# A classifier with CNN with the pretrained word2vec table
class MyCNN(nn.Module):
    def __init__(self, embedding, seq_size=DATA_MAX_LEN, cnn_out_ch=10, output_size=20):
        super(MyCNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.conv2 = nn.Conv1d(seq_size, cnn_out_ch, 2)
        self.conv3 = nn.Conv1d(seq_size, cnn_out_ch, 3)
        self.conv4 = nn.Conv1d(seq_size, cnn_out_ch, 4)
        self.conv5 = nn.Conv1d(seq_size, cnn_out_ch, 5)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.pool3 = nn.MaxPool1d(3, 2)
        self.pool4 = nn.MaxPool1d(4, 2)
        self.pool5 = nn.MaxPool1d(5, 2)
        out_size = lambda in_size, kernel, stride=1: np.floor((in_size - (kernel - 1) - 1) / stride + 1) # Assume) padding=0, dilation=1
        embed_size = embedding.shape[-1]
        union_size = int(out_size(out_size(embed_size, 2), 2, 2) + \
                         out_size(out_size(embed_size, 3), 3, 2) + \
                         out_size(out_size(embed_size, 4), 4, 2) + \
                         out_size(out_size(embed_size, 5), 5, 2)) * cnn_out_ch
        self.fc    = nn.Linear(union_size, output_size)

        self.embed.weight.requires_grad = False # Freeze 'self.embed'

    def forward(self, indices):
        v0 = self.embed(indices)
        x2 = self.pool2(F.relu(self.conv2(v0)))
        x3 = self.pool3(F.relu(self.conv3(v0)))
        x4 = self.pool4(F.relu(self.conv4(v0)))
        x5 = self.pool5(F.relu(self.conv5(v0)))
        union = torch.cat((x2, x3, x4, x5), dim=2)
        union = union.reshape(union.size(0), -1)
        out = self.fc(union)
        return out


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

    # 1.2. Load the word2vec lookup table
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    word2vec = torchtext.vocab.FastText('en')
    doc2index = lambda doc: [word2vec.stoi[token] for token in tokenizer(doc) if token in word2vec.stoi]
    print('* The size of vocabulary: ', word2vec.vectors.shape[0])

    # 1.3. Tensorize training and test data
    train_indices = [torch.LongTensor(doc2index(doc)[:DATA_MAX_LEN]) for doc in train_raw.data] # Trim if > DATA_MAX_LEN
    tests_indices = [torch.LongTensor(doc2index(doc)[:DATA_MAX_LEN]) for doc in tests_raw.data]
    train_tensors = nn.utils.rnn.pad_sequence(train_indices, batch_first=True)                  # Fill if < DATA_MAX_LEN
    tests_tensors = nn.utils.rnn.pad_sequence(tests_indices, batch_first=True)
    train_targets = torch.LongTensor(train_raw.target)
    tests_targets = torch.LongTensor(tests_raw.target)
    train_dloader = DataLoader(TensorDataset(train_tensors, train_targets), **DATA_LOADER_PARAM)
    tests_dloader = DataLoader(TensorDataset(tests_tensors, tests_targets), **DATA_LOADER_PARAM)

    # 2. Instantiate a model, loss function, and optimizer
    model = MyCNN(word2vec.vectors).to(dev)
    loss_func = F.cross_entropy
    optimizer = torch.optim.Adadelta(model.parameters(), **OPTIMIZER_PARAM)
    print('* The number of model parameters: ', sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))

    # 3. Train the model
    loss_list = []
    start = time.time()
    for epoch in range(1, EPOCH_MAX + 1):
        train_loss = train(model, train_dloader, loss_func, optimizer)
        valid_loss, valid_accuracy = evaluate(model, tests_dloader, loss_func)

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
