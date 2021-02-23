import torch
from torch import nn
from copy import deepcopy
from train_utils import Embeddings, PositionalEncoding
from interactor import Interactor
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
from utils import *
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math
from torch import nn

from torch import nn
import torch.nn.functional as F

from torch import nn
from train_utils import clones
from sublayer import LayerNorm, SublayerOutput

import torch
from torch import nn

from utils import *
from model import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch
from matplotlib import pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import copy
import math


import torch
from torchtext import data
from torchtext.vocab import Vectors
import pandas as pd
import numpy as np
import spacy
from sklearn.metrics import accuracy_score


class Config(object):
    N = 2
    d_model = 100
    d_ff = 512
    d_row = 60
    dropout = 0.1
    output_size = 1
    lr = 0.005
    max_epochs = 10
    batch_size = 64
    max_sen_len = 60
    pre_trained = False

class Matposer(nn.Module):
    def __init__(self, config, src_vocab, pre_trained=False):
        super(Matposer, self).__init__()
        self.config = config

        d_row, N, dropout = self.config.d_row, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff

        inter = Interactor(d_model, d_ff, out_row=d_row, dropout=dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(EncoderLayer(d_model, deepcopy(inter), deepcopy(ff), dropout), N)
        self.src_embed = nn.Sequential(
            Embeddings(d_model, src_vocab, pre_trained), deepcopy(position)
        )

        self.fc = nn.Linear(
            d_model,
            self.config.output_size
        )

        

    def forward(self, x):
        embedded_sents = self.src_embed(x.permute(1, 0)) # shape = (batch_size, sen_len, d_model)
        encoded_sents = self.encoder(embedded_sents)
        # final_feature_map = encoded_sents[:,-1,:]
        final_feature_map = torch.sum(encoded_sents,1)
        final_out = self.fc(final_feature_map)
        return final_out

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        # modify
        train_loss = 0
        val_loss = 0
        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                # y = (batch.label - 1).type(torch.cuda.LongTensor)
                y = (batch.label).type(torch.cuda.FloatTensor)
            else:
                x = batch.text
                # y = (batch.label - 1).type(torch.LongTensor)
                y = (batch.label).type(torch.FloatTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            
            #avg_train_loss = np.mean(losses)
            train_loss += loss.item()
            with torch.no_grad():
                val_accuracy = evaluate_model(self, val_iterator)
                val_loss += val_accuracy
            
            if i % 20 == 0:
                print("Iter: {}".format(i + 1))
                # avg_train_loss = np.mean(losses)
                
                # train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(train_loss/(i+1)))
                losses = []

                # Evalute Accuracy on validation set
                # val_accuracy = evaluate_model(self, val_iterator)
                
                print("\tVal RMSE: {:.4f}".format(val_accuracy/(i+1)))
                # self.train()
        train_losses.append(train_loss/len(train_iterator))
        val_accuracies.append(val_loss/len(val_iterator))
        return train_losses, val_accuracies

class Encoder(nn.Module):
    '''
    Matposer Encoder

    It is a stack of N layers.
    '''
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of Interactor and a feed forward layer
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, size, interactor, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.interactor = interactor
        self.feed_forward = feed_forward
        self.sublayer = SublayerOutput(size, dropout)
        self.size = size

    def forward(self, x):
        "Matposer Encoder"
        x = self.interactor(x)
        return self.sublayer(x, self.feed_forward)

class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Column_wise_nn(nn.Module):
    def __init__(self, d_row, d_column, d_ff, dropout=None):
        '''
        initialize column-wise neural network
        :param d_row: input row number
        :param d_ff: middle size row number
        :param dropout: default None
        '''
        super(Column_wise_nn, self).__init__()
        self.w_1 = nn.Linear(d_row, d_ff)
        self.w_2 = nn.Linear(d_ff, d_column)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = x.permute(0,2,1)
        d_k = x.size(-1)
        #output = self.w_2(self.dropout(F.relu(self.w_1(x)))) / math.sqrt(d_k)
        #output = F.softmax(output, dim=-1)
        output = self.w_2(self.dropout(F.relu(self.w_1(x)))) / math.sqrt(d_k)
        if self.dropout is not None:
            output = self.dropout(output)

        return output.permute(0,2,1)


class Row_wise_nn(nn.Module):
    def __init__(self, d_column, d_ff, out_row, dropout=None):
        super(Row_wise_nn, self).__init__()
        self.w_1 = nn.Linear(d_column, d_ff)
        self.w_2 = nn.Linear(d_ff, out_row)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        d_k = x.size(-1)
        output = self.w_2(self.dropout(F.relu(self.w_1(x)))) / math.sqrt(d_k)
        output = F.softmax(output, dim=-1)
        if self.dropout is not None:
            output = self.dropout(output)

        return output


class Interactor(nn.Module):
    def __init__(self, d_column, d_ff, out_row=30, dropout=0.1):
        '''
        :param d_row: dimension of output row number
        :param d_column: dimension of input column number
        :param d_ff: dimension of middle neural
        :param dropout: default 0.1
        '''
        super(Interactor, self).__init__()
        self.column_wise_nn = Column_wise_nn(out_row, d_column, d_ff, dropout)
        self.row_wise_nn = Row_wise_nn(d_column, d_ff, out_row, dropout)

    def forward(self, x):
        left_transposer = self.row_wise_nn(x)
        middle_term = torch.matmul(left_transposer.permute(0,2,1), x)
        output = self.column_wise_nn(middle_term)
        #output = torch.matmul(middle_term, right_transposer.permute(0,2,1))
        return output

class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    '''
    Usual Embedding layer with weights multiplied by sqrt(d_model)
    '''

    def __init__(self, d_model, vocab, pre_trained=False):
        super(Embeddings, self).__init__()
        if pre_trained is False:
            self.lut = nn.Embedding(len(vocab), d_model)
        else:
            self.lut = nn.Embedding.from_pretrained(vocab)
        self.d_model = d_model

        # self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe[:, 1::2] = torch.cos(
            torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))  # torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def get_embedding_matrix(vocab_chars):
    # return one hot emdding
    vocabulary_size = len(vocab_chars)
    onehot_matrix = np.eye(vocabulary_size, vocabulary_size)
    return onehot_matrix


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}

    # def parse_label(self, label):
    #     '''
    #     Get the actual labels from label string
    #     Input:
    #         label (string) : labels of the form '__label__2'
    #     Returns:
    #         label (int) : integer value corresponding to label string
    #     '''
    #     if not isinstance(label, str):
    #         raise Exception(
    #             'type of label should be str. The type of label was {}'.format(
    #                 type(label)))
    #
    #     return int(label.strip()[-1])

    # def get_pandas_df(self, filename):
    #     '''
    #     Load the data into Pandas.DataFrame object
    #     This will be used to convert data to torchtext object
    #     '''
    #     with open(filename, 'r') as datafile:
    #         data = [line.strip().split(',', maxsplit=1) for line in datafile]
    #         data_text = list(map(lambda x: x[1], data))
    #         data_label = list(map(lambda x: self.parse_label(x[0]), data))
    #
    #     full_df = pd.DataFrame({"text": data_text, "label": data_label})
    #     return full_df

    def load_data(self, train_file, test_file, val_file=None, pre_trained=False):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        Inputs:
            train_file (String): absolute path to training file
            test_file (String): absolute path to test file
            val_file (String): absolute path to validation file
        '''
        # Loading Tokenizer
        NLP = spacy.load('en')

        def tokenizer(sent):
            return list(
                x.text for x in NLP.tokenizer(sent) if x.text != " ")

        # Creating Filed for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = pd.read_csv(train_file)
        train_df = train_df[['original', 'meanGrade']]
        train_df = train_df.rename(columns={'original': "text",
                                            'meanGrade': 'label'})
        train_examples = [
            data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)

        test_df = pd.read_csv(test_file)
        test_df = test_df[['original', 'meanGrade']]
        test_df = test_df.rename(columns={'original': "text",
                                          'meanGrade': 'label'})
        # test_df = self.get_pandas_df(test_file)
        test_examples = [
            data.Example.fromlist(
                i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)

        # If validation file exists, load it. Otherwise get validation data
        # from training data
        if val_file:
            # val_df = self.get_pandas_df(val_file)
            val_df = pd.read_csv(val_file)
            val_df = val_df[['original', 'meanGrade']]
            val_df = val_df.rename(columns={'original': "text",
                                            'meanGrade': 'label'})
            val_examples = [
                data.Example.fromlist(
                    i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)

        if pre_trained:
            TEXT.build_vocab(train_data, vectors='glove.6B.100d')
            self.vocab = TEXT.vocab.vectors
        else:
            TEXT.build_vocab(train_data)
            self.vocab = TEXT.vocab

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True
        )

        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False
        )

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} test examples".format(len(test_data)))
        print("Loaded {} validation examples".format(len(val_data)))


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        # predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        predicted = y_pred.cpu().data
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())

    # score = accuracy_score(all_y, np.array(all_preds).flatten())
    return np.sqrt(((all_y - np.array(all_preds)) ** 2).mean())


if __name__ == '__main__':
    torch.cuda.empty_cache()
    config = Config
    train_file = './data/train.csv'
    test_file = './data/test.csv'

    val_file = './data/dev.csv'
    dataset = Dataset(config)
    dataset.load_data(train_file, test_file, val_file, config.pre_trained)
    # print(dataset.vocab)
    model = Matposer(config, dataset.vocab, config.pre_trained)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)


    def RMSELoss(yhat, y):
        return torch.sqrt(torch.mean((yhat - y) ** 2))


    loss = RMSELoss
    model.add_optimizer(optimizer)
    model.add_loss_op(loss)

    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    plt.figure()
    x = np.arange(1, config.max_epochs + 1)

    y1 = train_losses
    y2 = val_accuracies
    plt.plot(x, y1, label='Train loss')
    plt.plot(x, y2, label='Validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('pre-trained representations')
    plt.xticks(np.arange(1, config.max_epochs + 1))
    plt.legend()
    # plt.ylim((0,0.5))
    plt.savefig('/content/pre_trained.jpg')

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)

    print('Final Training RMSE: {:.4f}'.format(train_acc))
    print('Final Validation RMSE: {:.4f}'.format(val_acc))
    print('Final Test RMSE: {:.4f}'.format(test_acc))
