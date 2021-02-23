from utils import *
from model import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch
from matplotlib import pyplot as plt
import numpy as np

if __name__=='__main__':
    torch.cuda.empty_cache()
    config = Config
    train_file = './data/train.csv'
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    test_file = './data/test.csv'
    
    if len(sys.argv) > 3:
        test_file = sys.argv[2]
    val_file = './data/dev.csv'
    dataset = Dataset(config)
    dataset.load_data(train_file, test_file, val_file, config.pre_trained)
    # print(dataset.vocab)
    model = Matposer(config, dataset.vocab, config.pre_trained)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    def RMSELoss(yhat,y): return torch.sqrt(torch.mean((yhat-y)**2)) 

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
    x = np.arange(1, config.max_epochs+1)

    y1 = train_losses
    y2 = val_accuracies
    plt.plot(x, y1, label='Train loss')
    plt.plot(x, y2, label='Validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('pre-trained representations')
    plt.xticks(np.arange(1, config.max_epochs+1))
    plt.legend()
    # plt.ylim((0,0.5))
    plt.savefig('/content/pre_trained.jpg')
    
    
    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)

    print('Final Training RMSE: {:.4f}'.format(train_acc))
    print('Final Validation RMSE: {:.4f}'.format(val_acc))
    print('Final Test RMSE: {:.4f}'.format(test_acc))


