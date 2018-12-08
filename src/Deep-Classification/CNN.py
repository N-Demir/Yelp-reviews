import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
from pathlib import Path
from datetime import datetime
import numpy as np
import os
import random
from sklearn.metrics import precision_recall_fscore_support


N_EPOCHS = 2000
BATCH_SIZE = 64
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
EPOCH_SAVE = 5
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
REPOSITORY_NAME = 'Yelp-reviews'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#SEED = 1234
#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True

#TEXT = data.Field(tokenize='spacy')
# LABEL = data.LabelField(dtype=torch.float)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        #x = [sent len, batch size]

        x = x.permute(1, 0)

        #x = [batch size, sent len]

        embedded = self.embedding(x)

        #embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        #embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        #pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        #cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


# def binary_accuracy(preds, y):
#     """
#     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#     """
#
#     #round predictions to the closest integer
#     rounded_preds = torch.round(torch.sigmoid(preds))
#     correct = (rounded_preds == y).float() #convert into float for division
#     acc = correct.sum()/len(correct)
#     return acc

def batch_accuracy(logits, y):
	# Pass through sigmoid with decision bound at 0.5
	predictions = torch.round(torch.sigmoid(logits))
	correct = (predictions == y).float()
	return correct.sum() / len(correct)

def batch_precision_recall_f_score(preds, y):
    y_pred = torch.round(torch.sigmoid(preds))
    y_pred = y_pred.detach().numpy()
    y_true = y.detach().numpy()
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1_score

def train(model, iterator, optimizer, loss_function):

    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0
    epoch_recall = 0
    epoch_f_score = 0

    model.train()

    for batch_i, batch in enumerate(iterator):
        print ("Starting Batch: %d" % batch_i)

        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = loss_function(predictions, batch.label)

        acc = batch_accuracy(predictions, batch.label)
        precision, recall, f1_score = batch_precision_recall_f_score(predictions, batch.label)

        print ("Batch %d accuracy: %f" % (batch_i, acc))

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_prec += precision
        epoch_recall += recall
        epoch_f_score += f1_score

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_prec / len(iterator), epoch_recall / len(iterator), epoch_f_score / len(iterator)

def evaluate(model, iterator, loss_function):

    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0
    epoch_recall = 0
    epoch_f_score = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            predictions = model(batch.text).squeeze(1)

            loss = loss_function(predictions, batch.label)

            acc = batch_accuracy(predictions, batch.label)

            precision, recall, f1_score = batch_precision_recall_f_score(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_prec += precision
            epoch_recall += recall
            epoch_f_score += f1_score

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_prec / len(iterator), epoch_recall / len(iterator), epoch_f_score / len(iterator) 

def main():
    # load the dataset
    TEXT, train_itr, valid_itr, test_itr = data_loader.load_data()
    # train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    # train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    #
    # TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
    # LABEL.build_vocab(train_data)

    input_dim = len(TEXT.vocab)

    # train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    #     (train_data, valid_data, test_data),
    #     batch_size=BATCH_SIZE,
    #     device=device)

    # create an instance of the model
    model = CNN(vocab_size=input_dim, embedding_dim=EMBEDDING_DIM, n_filters=N_FILTERS, filter_sizes=FILTER_SIZES, output_dim=OUTPUT_DIM, dropout=DROPOUT)

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    optimizer = optim.Adam(model.parameters())

    loss_function = nn.BCEWithLogitsLoss()

    # allow for running on GPU
    model = model.to(device)
    loss_function = loss_function.to(device)

    for epoch in range(N_EPOCHS):
        
        train_loss, train_acc, train_prec, train_recall, train_f_score = train(model, train_itr, optimizer, loss_function)
        valid_loss, valid_acc, valid_prec, valid_recall, valid_f_score = evaluate(model, valid_itr, loss_function)
        
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}| Train Acc: {train_acc*100:.2f}% | Train f1_score: {train_f_score*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | Val. f1_score: {valid_f_score*100:.2f}%')
        saveMetrics('train', train_acc, train_loss, train_prec, train_recall, train_f_score, epoch)
        saveMetrics('valid', valid_acc, valid_loss, valid_prec, valid_recall, valid_f_score, epoch)
        if (epoch % EPOCH_SAVE == 0):
            saveModel(model, epoch)

    saveModel(model, epoch)

def setupCheckpoints():
    def get_repository_path():
        """
        Returns the path of the project repository

        Uses the global REPOSITORY_NAME constant and searches through parent directories
        """
        p = Path(__file__).absolute().parents
        for parent in p:
            if parent.name == REPOSITORY_NAME:
                return parent

    p = get_repository_path()
    checkpoints_folder = p / 'checkpoints'
    LSTM_folder = checkpoints_folder / 'CNN'
    cur_folder = LSTM_folder / CURRENT_TIME


    checkpoints_folder.mkdir(exist_ok=True)
    LSTM_folder.mkdir(exist_ok=True)
    cur_folder.mkdir(exist_ok=True)

    return cur_folder

def saveModel(model, epoch):
    path = setupCheckpoints()
    model_folder = path / 'models'
    model_folder.mkdir(exist_ok=True)

    model_path = model_folder / '{:02d}-model.pt'.format(epoch)

    state = model.state_dict()
    torch.save(state, model_path)

def saveMetrics(prefix, accuracy, loss, precision, recall, f1_score, epoch):
    path = setupCheckpoints()

    accuracy_path = path / '{}-accuracy.txt'.format(prefix)
    loss_path = path / '{}-loss.txt'.format(prefix)
    precision_path = path / '{}-precision.txt'.format(prefix)
    recall_path = path / '{}-recall.txt'.format(prefix)
    f1_score_path = path / '{}-f1_score.txt'.format(prefix)

    with open(accuracy_path, 'a+') as f:
        f.write('{},{}\n'.format(epoch, accuracy))
    with open(loss_path, 'a+') as f:
        f.write('{},{}\n'.format(epoch, loss))
    with open(precision_path, 'a+') as f:
        f.write('{},{}\n'.format(epoch, precision))
    with open(recall_path, 'a+') as f:
        f.write('{},{}\n'.format(epoch, recall))
    with open(f1_score_path, 'a+') as f:
        f.write('{},{}\n'.format(epoch, f1_score))

def saveAccuracyAndLoss(prefix, accuracy, loss, epoch):
    path = setupCheckpoints()

    accuracy_path = path / '{}-accuracy.txt'.format(prefix)
    loss_path = path / '{}-loss.txt'.format(prefix)

    with open(accuracy_path, 'a+') as f:
        f.write('{},{}\n'.format(epoch, accuracy))
    with open(loss_path, 'a+') as f:
        f.write('{},{}\n'.format(epoch, loss))


if __name__ == '__main__':
    main()
