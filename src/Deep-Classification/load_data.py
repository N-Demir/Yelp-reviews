import spacy
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import tqdm

spacy_en = spacy.load('en') # english language model

def tokenizer(text): # create a tokenizer function
	return [tok.text for tok in spacy_en.tokenizer(text)]

# Fields for our dataset
TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = Field(sequential=False, use_vocab=False) # We have already converted to 0 / 1


def get_data_set():
	# For now we just grab train to see what it looks like
	train = TabularDataset(
        path='../data/op_spam_v1.4/labeled_reviews.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])

	return train

def get_iterator(dataset):
	dataset_iter = Iterator(
			dataset, # we pass in the datasets we want the iterator to draw data from
			batch_size=512,
			device=-1, # if you want to use the GPU, specify the GPU number here
			sort_key=lambda x: len(x.Text), # the BucketIterator needs to be told what function it should use to group the data.
			sort_within_batch=False,
			repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
		)
	return dataset_iter
	#Iterator(tst, batch_size=64, device=-1, sort=False, sort_within_batch=False, repeat=False)


# Test Model
class LSTMBaseline(nn.Module):
	def __init__(self, hidden_dim, emb_dim=300, num_linear=1):
		super().__init__() # don't forget to call this!
		self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
		self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
		self.linear_layers = []
		for _ in range(num_linear - 1):
			self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
			self.linear_layers = nn.ModuleList(self.linear_layers)
		self.predictor = nn.Linear(hidden_dim, 1)

	'''
	def forward(self, sentence):
		# May want to init the hidden in the forward pass??
		embeddings = self.word_embeddings(sentence) # shape - (batch, seq_len, embed_size)
		# Permute embeddings to follow standard lstm shape - (seq_len, batch, embed_size)
		embeddings = embeddings.permute(1, 0, 2)

		lstm_out, self.hidden = self.lstm(embeddings, self.hidden)

		# Pass final hidden state through the linear layer to get predicition
		# NOTE: we want the hidden layer of the last LSTM layer so we use -1
		output = self.label(self.hidden[0][-1]) # hidden_state - (num_layers, batch_size, hidden_size)

		return output
	'''

	def forward(self, seq):
		print (seq.shape)
		hdn, _ = self.encoder(self.embedding(seq))
		feature = hdn[-1, :, :]
		for layer in self.linear_layers:
			feature = layer(feature)
		preds = self.predictor(feature)
		return preds



def main():
	train = get_data_set()

	# Load the embeddings -- later
	TEXT.build_vocab(train)

	train_itr = get_iterator(train)
	em_sz = 100
	nh = 500
	nl = 1
	model = LSTMBaseline(nh, em_sz, nl)

	opt = optim.Adam(model.parameters(), lr=1e-2)
	loss_func = nn.BCEWithLogitsLoss()

	epochs = 2

	for epoch in range(1, epochs + 1):
		running_loss = 0.0
		running_corrects = 0
		model.train() # turn on training mode
		for data in train_itr:
			x = data.Text
			#print (x)
			y = torch.autograd.Variable(data.Label, requires_grad=False).float()
			#y = torch.autograd.Variable(data.Label,requires_grad=False).long()
		#for x, y in tqdm.tqdm(train_dl): # thanks to our wrapper, we can intuitively iterate over our data!
			opt.zero_grad()

			preds = model(x)
			loss = loss_func(preds.squeeze(1), y)
			loss.backward()
			opt.step()

			running_loss += loss.data[0] * x.size(0)

		epoch_loss = running_loss / len(train_itr)
		print (epoch_loss)

		'''
		# calculate the validation loss for this epoch
		val_loss = 0.0
		model.eval() # turn on evaluation mode
		for x, y in valid_dl:
			preds = model(x)
			loss = loss_func(y, preds)
			val_loss += loss.data[0] * x.size(0)

		val_loss /= len(vld)
		'''


if __name__ == '__main__':
	main()