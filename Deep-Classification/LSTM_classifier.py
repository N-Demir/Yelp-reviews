import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, vocab_size, word_vec_weights, output_size=1, hidden_size=256, 
                embedding_dim=100, num_layers=1, dropout=0, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        """
        Arguments
        ---------
        batch_size : Size of the batch
        output_size : 1 - will pass then through binary-cross-entropy with sigmoid in loss
        hidden_size : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words

        Note we will probably pre-process the data to get GloVe vectors rather than the actual words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        num_cells: Specifies the number of LSTM cells to stack
        dropout: The drop out ratio - default to 0 meaning no dropout
        
        """
        
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        
        # Initializing the look-up table to be the pre-trained Glove vecs
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        #self.word_embeddings.weight = nn.Parameter(word_vec_weights, requires_grad=False) 

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=bidirectional)

        # Linear layer for the classification given final hidden state
        # Interesting idea could be chaining linear layers
        # In the case of bidirectional lstm we pass h_f from the
        # the forward and backward direction
        linear_input = hidden_size * self.num_directions

        self.label = nn.Linear(linear_input, output_size)

        # Create Dropout Layer that can be used on the
        # output of layers where we want to add dropout 
        self.dropout = nn.Dropout(dropout)

        # Initialize the initial hidden state
        # Tuple - (h_0, c_0)
        self.hidden_state = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state. (h0, c0)
        # Hidden layer dimension - num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                    torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, sentence):
        # May want to init the hidden in the forward pass??
        embeddings = self.word_embeddings(sentence) # - (batch, seq_len, embed_size)
       
        # No need I guess
        # Permute embeddings to follow standard lstm shape - (seq_len, batch, embed_size)
        #embeddings = embeddings.permute(1, 0, 2)
        embeddings = self.dropout(embeddings)

        #lstm_out, self.hidden_state = self.lstm(embeddings, self.hidden_state)
        # So we don't have to deal with initializing hidden states
        lstm_out, (hidden, cell) = self.lstm(embeddings)
        print (hidden.shape)

        # Re=shape hidden
        #hidden = hidden.view(self.num_layers, self.num_directions, self.batch_size, self.hidden_size)
        #print (hidden.shape)


        # Get the hidden state of the last lstm layer for the forward direction
        #linear_input = hidden[-1, 0, :, :]
        # If we have bidirectional layer than we concatenate the directions
        #if (self.num_directions == 2):
            #print ("here")
            #linear_input = torch.cat((hidden[-1, 0, :, :], hidden[-1, 1, :, :]), dim=1)

        #linear_input = self.dropout(linear_input)
        # Test!
        linear_input = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
         
        # Pass final hidden state through the linear layer to get predicition
        # NOTE: we want the hidden layer of the last LSTM layer so we use -1
        #output = self.label(self.hidden_state[0][-1]) # hidden_state - (num_layers, batch_size, hidden_size)
        # Note we squeeze if the batch_size = 1
        output = self.label(linear_input.squeeze(0))

        return output # size - (batch_size, 1)




