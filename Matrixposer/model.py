import torch
from torch import nn
from copy import deepcopy
from train_utils import Embeddings, PositionalEncoding
from interactor import Interactor
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
from utils import *

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
            
            avg_train_loss = np.mean(losses)
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