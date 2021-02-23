import torch
from torchtext import data
from torchtext.vocab import Vectors
import pandas as pd
import numpy as np
import spacy
from sklearn.metrics import accuracy_score



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

    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        if not isinstance(label, str):
            raise Exception(
                'type of label should be str. The type of label was {}'.format(
                    type(label)))

        return int(label.strip()[-1])

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        with open(filename, 'r') as datafile:
            data = [line.strip().split(',', maxsplit=1) for line in datafile]
            data_text = list(map(lambda x: x[1], data))
            data_label = list(map(lambda x: self.parse_label(x[0]), data))

        full_df = pd.DataFrame({"text": data_text, "label": data_label})
        return full_df

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

        def tokenizer(sent): return list(
            x.text for x in NLP.tokenizer(sent) if x.text != " ")

        # Creating Filed for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = pd.read_csv(train_file)
        train_df = train_df[['original','meanGrade']]
        train_df = train_df.rename(columns={'original': "text",
                                      'meanGrade': 'label'})
        train_examples = [
            data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)

        test_df = pd.read_csv(test_file)
        test_df = test_df[['original','meanGrade']]
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
            val_df = val_df[['original','meanGrade']]
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

        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))
        print ("Loaded {} validation examples".format(len(val_data)))


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
