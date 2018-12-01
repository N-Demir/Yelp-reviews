import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from LSTM_classifier import LSTMClassifier
import data_loader

EMBEDDING_DIM = 100
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 1000
GRAD_CLIP = 1e-1
NUM_LAYERS = 2

# Should include gradient clipping!!

def train(model, iterator, loss_function, optimizer):
	
	epoch_loss = 0
	epoch_accuracy = 0

	model.train()
	for batch in iterator:
		# Get the reviews
		reviews = batch.Text
		# Get the labels
		labels = batch.Label

		optimizer.zero_grad()
		# Re-set the hidden state for the LSTM
		model.hidden_state = model.init_hidden()

		# Forward pass
		logits = model.forward(reviews) # - (batch, 1)
		logits = logits.squeeze(1)

		# Compute loss
		loss = loss_function(logits, labels)
		# Maybe compute accuracy for later
		accuracy = batch_accuracy(logits, labels)

		loss.backward()
		# Clip gradients
		torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

		optimizer.step()

		epoch_loss += loss.item()
		epoch_accuracy += accuracy.item()

	# This gives average loss maybe on the batch
	return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def batch_accuracy(logits, y):
	# Pass through sigmoid with decision bound at 0.5
	predictions = torch.round(torch.sigmoid(logits))
	correct = (predictions == y).float()
	return correct.sum() / len(correct)


def main():
	# Load the data-set
	TEXT, train_itr = data_loader.load_data()

	# Extract usefull features from the text dataset object
	vocab_size = len(TEXT.vocab)
	glove_vec_weights = TEXT.vocab.vectors

	# Note we need to add weights for embeddings in a bit -- maybe for initial test don't have learned embeddings??
	model = LSTMClassifier(batch_size=BATCH_SIZE, hidden_size=HIDDEN_DIM, 
		vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, word_vec_weights=glove_vec_weights)


	# Let us train this baby
	loss_function = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters())
	for epoch in range(EPOCHS):
		print ("Epoch....%d" % (epoch))
		train_loss, train_accuracy = train(model, train_itr, loss_function, optimizer)

		print("Training loss: %f" % (train_loss))
		print("Avg batch accuracy: %f" % (train_accuracy))

	


if __name__ == '__main__':
	main()


