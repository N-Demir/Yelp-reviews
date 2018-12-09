import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from LSTM_classifier import LSTMClassifier
import data_loader
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support


EPOCH_SAVE = 5
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 2000
GRAD_CLIP = 1e-1
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
CLIP_GRADIENTS = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
REPOSITORY_NAME = 'Yelp-reviews'

# If we want to reproduce something very specific
#SEED = 1234
#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True

def train(model, iterator, loss_function, optimizer):
	
	epoch_loss = 0.
	epoch_acc = 0.
	epoch_real_prec = 0.
	epoch_fake_prec = 0.
	epoch_real_recall = 0.
	epoch_fake_recall = 0.
	epoch_real_f_score = 0.
	epoch_fake_f_score = 0.

	model.train()
	print (len(iterator))
	for batch_idx, batch in enumerate(iterator):
		print ("Starting Batch: %d" % batch_idx)

		# Get the reviews
		reviews = batch.text 
		labels = batch.label 

		optimizer.zero_grad()

		# Forward pass
		logits = model.forward(reviews) 
		# logits = [batch, 1]

		logits = logits.squeeze(1)
		# logits = [batch]

		# Compute loss
		loss = loss_function(logits, labels)
		# Maybe compute accuracy for later
		accuracy = batch_accuracy(logits, labels)
		precisions, recalls, f1_scores = batch_precision_recall_f_score(logits, batch.label)
		print ("Batch %d accuracy: %f" % (batch_idx, accuracy))

		loss.backward()
		# Clip gradients 
		#torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

		optimizer.step()

		epoch_loss += loss.item()
		epoch_acc += accuracy.item()
		epoch_real_prec += precisions[0]
		epoch_fake_prec += precisions[1]
		epoch_real_recall += recalls[0]
		epoch_fake_recall += recalls[1]
		epoch_real_f_score += f1_scores[0]
		epoch_fake_f_score += f1_scores[1]

	# This gives average loss maybe on the batch
	avg_prec = [epoch_real_prec / len(iterator), epoch_fake_prec / len(iterator)]
	avg_recall = [epoch_real_recall / len(iterator),  epoch_fake_recall / len(iterator)]
	avg_f_score = [epoch_real_f_score / len(iterator),  epoch_fake_f_score / len(iterator)]
	return epoch_loss / len(iterator), epoch_acc / len(iterator), avg_prec, avg_recall, avg_f_score

def evaluate_model(model, iterator, loss_function):
	epoch_loss = 0.
	epoch_acc = 0.
	epoch_real_prec = 0.
	epoch_fake_prec = 0.
	epoch_real_recall = 0.
	epoch_fake_recall = 0.
	epoch_real_f_score = 0.
	epoch_fake_f_score = 0.

	model.eval()
	with torch.no_grad():
		for batch_idx, batch in enumerate(iterator):

			reviews = batch.text
			labels = batch.label 

			logits = model.forward(reviews)
			logits = logits.squeeze(1)

			loss = loss_function(logits, labels)
			accuracy = batch_accuracy(logits, labels)
			precisions, recalls, f1_scores = batch_precision_recall_f_score(logits, batch.label)

			epoch_loss += loss.item()
			epoch_acc += accuracy.item()
			epoch_real_prec += precisions[0]
			epoch_fake_prec += precisions[1]
			epoch_real_recall += recalls[0]
			epoch_fake_recall += recalls[1]
			epoch_real_f_score += f1_scores[0]
			epoch_fake_f_score += f1_scores[1]

	# May want to do this differently - prob doesnt matter really
	avg_prec = [epoch_real_prec / len(iterator), epoch_fake_prec / len(iterator)]
	avg_recall = [epoch_real_recall / len(iterator),  epoch_fake_recall / len(iterator)]
	avg_f_score = [epoch_real_f_score / len(iterator),  epoch_fake_f_score / len(iterator)]
	return epoch_loss / len(iterator), epoch_acc / len(iterator), avg_prec, avg_recall, avg_f_score 

def batch_accuracy(logits, y):
	# Pass through sigmoid with decision bound at 0.5
	predictions = torch.round(torch.sigmoid(logits))
	correct = (predictions == y).float()
	return correct.sum() / len(correct)

def batch_precision_recall_f_score(preds, y):
    y_pred = torch.round(torch.sigmoid(preds))
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y.detach().cpu().numpy()
    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred)
    # In the case that y passed in is all one class, then we still want to return array with 2 elems
    if len(precisions) != 2:
        # If this happens because y is not all of one type then we have an issueeee
        for a in y:
            if y[0] != a: raise Exception('WHAT THE FUUUUUUCK')
        if y[0] == 0:
            return precisions + [0.], recalls + [0.], f1_scores + [0.]
        else:
            return [0.] + precisions, [0.] + recalls, [0.] + f1_scores
    return precisions, recalls, f1_scores


def main():
	# Load the data-set
	TEXT, train_itr, valid_itr, test_itr = data_loader.load_data()

	# Extract usefull features from the text dataset object
	vocab_size = len(TEXT.vocab)
	glove_vec_weights = TEXT.vocab.vectors

	# Note we need to add weights for embeddings in a bit -- maybe for initial test don't have learned embeddings??
	model = LSTMClassifier(batch_size=BATCH_SIZE, hidden_size=HIDDEN_DIM, 
		vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, word_vec_weights=glove_vec_weights, 
		num_layers=NUM_LAYERS, dropout=DROPOUT, bidirectional=BIDIRECTIONAL)

	model.word_embeddings.weight.data.copy_(glove_vec_weights)

	# Let us train this baby
	optimizer = optim.Adam(model.parameters())
	loss_function = nn.BCEWithLogitsLoss()

	# Allow for running on GPU
	model = model.to(DEVICE)
	loss_function = loss_function.to(DEVICE)

	for epoch in range(1, EPOCHS):
		print ("Epoch....%d" % (epoch))
		#train_loss, train_accuracy = train(model, train_itr, loss_function, optimizer)
		#val_loss, val_accuracy = evaluate_model(model, valid_itr, loss_function)

		train_loss, train_acc, train_prec, train_recall, train_f_score = train(model, train_itr, loss_function, optimizer)
		valid_loss, valid_acc, valid_prec, valid_recall, valid_f_score = evaluate_model(model, valid_itr, loss_function)

		## Here you have accuracy to the losses that we want to save

		print(f'| Epoch: {epoch:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

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
	LSTM_folder = checkpoints_folder / 'LSTM'
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

def saveMetrics(prefix, accuracy, loss, precisions, recalls, f1_scores, epoch):
    real_precision, fake_precision = precisions
    real_recall, fake_recall = recalls
    real_f1_score, fake_f1_score = f1_scores

    path = setupCheckpoints()

    accuracy_path = path / '{}-accuracy.txt'.format(prefix)
    loss_path = path / '{}-loss.txt'.format(prefix)

    real_precision_path = path / '{}-real-precision.txt'.format(prefix)
    fake_precision_path = path / '{}-fake-precision.txt'.format(prefix)
    real_recall_path = path / '{}-real-recall.txt'.format(prefix)
    fake_recall_path = path / '{}-fake-recall.txt'.format(prefix)
    real_f1_score_path = path / '{}-real-f1_score.txt'.format(prefix)
    fake_f1_score_path = path / '{}-fake-f1_score.txt'.format(prefix)

    def writeMetric(metric_path, value):
        with open(metric_path, 'a+') as f:
            f.write('{},{}\n'.format(epoch, value))

    writeMetric(accuracy_path, accuracy)
    writeMetric(loss_path, loss)
    writeMetric(real_precision_path, real_precision)
    writeMetric(fake_precision_path, fake_precision)
    writeMetric(real_recall_path, real_recall)
    writeMetric(fake_recall_path, fake_recall)
    writeMetric(real_f1_score_path, real_f1_score)
    writeMetric(fake_f1_score_path, fake_f1_score)

if __name__ == '__main__':
	main()
