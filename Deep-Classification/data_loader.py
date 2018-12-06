import spacy
from torchtext.data import Field, LabelField
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from torchtext import datasets
import torch
from torch.nn import functional as F

spacy_en = spacy.load('en') # english language model
# Later we will have a field for the actual folder 
# and a path for test and train files
train_path = '../data/op_spam_v1.4/labeled_reviews.tsv'
BATCH_SIZE = 64

# Do this for testing 
# To see if we match the results from online
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Note that now everything is tsv but would like json!!
def load_data():
    # Fields for the dataset
    # The actual review message
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = LabelField(dtype=torch.float)

    # Get the train dataset for now
    '''
    train_data = TabularDataset(
        path=train_path, format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])
    '''
    # Try loading in the IMB dataset to label pos or negative
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL) 

    # Now we need to build the vocab for our actual data
    # Here we will use the pre-trained word vetors from "glove.6b.100"
    # We have to see if we need to put this file in the same folder?
    TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
    # Not totally sure why we do this here
    LABEL.build_vocab(train_data) # Probably do not need this

    # Print stuff for sanity checks
    print ('Size of the vocab: ' + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    # This seems very strange
    print ("Label Length: " + str(len(LABEL.vocab)))

    # Get the corresponding iterator for the train dataset - again want to later do for all
    train_itr = Iterator(train_data, batch_size=BATCH_SIZE, device=-1, sort_key=lambda x: len(x.Text), 
            sort_within_batch=False, repeat=False, shuffle=True
        )

    return TEXT, train_itr




