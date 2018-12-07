import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from LSTM_classifier import LSTMClassifier
import data_loader

EPOCH_SAVE = 10
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 2
GRAD_CLIP = 1e-1
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# Should include gradient clipping!!

def train(model, iterator, loss_function, optimizer):
	
	epoch_loss = 0
	epoch_accuracy = 0

	model.train()
	print (len(iterator))
	for batch_idx, batch in enumerate(iterator):
		print ("Starting Batch: %d" % batch_idx)

		# Get the reviews
		#reviews = batch.Text
		reviews = batch.text # For testing data set
		# Get the labels
		#labels = batch.Label
		labels = batch.label # For testing

		optimizer.zero_grad()
		# Re-set the hidden state for the LSTM
		#model.hidden_state = model.init_hidden()

		# Forward pass
		logits = model.forward(reviews) 
		# logits = [batch, 1]

		logits = logits.squeeze(1)
		# logits = [batch]

		# Compute loss
		loss = loss_function(logits, labels)
		# Maybe compute accuracy for later
		accuracy = batch_accuracy(logits, labels)
		print ("Batch %d accuracy: %f" % (batch_idx, accuracy))

		loss.backward()
		# Clip gradients
		#torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

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

	# May not need?
	for epoch in range(EPOCHS):
		print ("Epoch....%d" % (epoch))
		train_loss, train_accuracy = train(model, train_itr, loss_function, optimizer)

		print("Training loss: %f" % (train_loss))
		print("Avg batch accuracy: %f" % (train_accuracy))
		if (epoch % EPOCH_SAVE == 0):
			save_model_by_name(model, epoch + 1)

	# Save one truely at the end
	if ((EPOCHS - 1) % EPOCH_SAVE != 0) :
		save_model_by_name(model, epoch + 1)



def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Change for now
    #file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    file_path = os.path.join(save_dir, 'model-{:02d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))
	


if __name__ == '__main__':
	main()


