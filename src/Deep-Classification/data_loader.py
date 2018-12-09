import spacy
from torchtext.data import Field, LabelField
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from torchtext import datasets
import torch
from torch.nn import functional as F
import random

# English language model
# Though we should play potentially with different
# language models
spacy_en = spacy.load('en')

path = '../../data/YelpChi/labeled_reviews_balanced.tsv'
BATCH_SIZE = 64

# Do this for testing
# To see if we match the results from online
#SEED = 1234
SEED = 229
TRAIN_VAL_TEST_SPLIT = [0.9, 0.05, 0.05]
TRAIN_SPLIT = 0.9
VAL_TEST_SPLIT = 0.5

# Sets the random number generator of torch
#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
# May want to play with this for reproducability
#torch.backends.cudnn.deterministic = True

def tokenizer(text): # create a tokenizer function
    # Consider using tok.lemma to lemmatize the
    # vocabulary rather than true words

    # We can also consider removing stop words!!

    # Consider appending potentially named entity tags??
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Note that now everything is tsv but would like json!!
def load_data(preprocessing=None):
    # Fields for the dataset
    # The actual review message

    #TEXT = Field(tokenize='spacy') # -- Old way, unclear exactly what language model is used
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True, preprocessing=preprocessing)
    LABEL = LabelField(dtype=torch.float)

    # Get the entire dataset that we will then split
    data = TabularDataset(
        path=path, format='tsv',
        fields=[('text', TEXT), ('label', LABEL)])

    # We should probabily look at the proportion of fake to non fake in each of these
    # set to make sure it is fairly even. Though probabilistically it should be I suppose
    train_data, valid_data, test_data = data.split(split_ratio=TRAIN_VAL_TEST_SPLIT, random_state=random.seed(SEED))
    #valid_data, test_data = test_data.split(split_ratio=VAL_TEST_SPLIT, random_state=random.seed(SEED))

    print ('Size of train set: ' + str(len(train_data.examples)))
    print ('Size of val / test: ' + str(len(valid_data.examples)))

    '''
    # Try loading in the IMB dataset to label pos or negative
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    # Get train/valid split!!
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    '''

    # Now we need to build the vocab for our actual data
    # Here we will use the pre-trained word vetors from "glove.6b.100"
    TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
    LABEL.build_vocab(train_data)

    # Print stuff for sanity checks
    print ('Size of the vocab: ' + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_itr, valid_itr, test_itr = BucketIterator.splits((train_data, valid_data, test_data),
        batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.text))

    return TEXT, train_itr, valid_itr, test_itr
