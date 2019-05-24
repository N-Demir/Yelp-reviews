import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import csv
import util as ut
import spacy
from sklearn import preprocessing
from random import shuffle
#from gensim.models.ldamodel import LdaModel

spacy_en = spacy.load('en')

min_max_scaler = preprocessing.MinMaxScaler()

# Get the reviews from the chicago dataset
#reviews, labels = ut.load_yelp_dataset_full("../data/YelpChi/")
reviews, labels = ut.load_tsv_dataset("../data/large_balanced_hotel.tsv")
print (reviews.shape)

zipped = list(zip(reviews, labels))

shuffle(zipped)

reviews, labels = zip(*zipped)


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(reviews, labels)

# Let us get the tf-idf for the reviews
#tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100, ngram_range=(1, 2))
# This is actually counts!!!!
#tfidf_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000, ngram_range=(1), stop_words='english')
unigram_count = CountVectorizer(stop_words='english')
bigram_count = CountVectorizer(ngram_range=(2, 2), stop_words='english')

#tfidf_vect.fit(reviews)
unigram_count.fit(reviews)
bigram_count.fit(reviews)

# Trasform
unigram_counts = unigram_count.transform(reviews)
bigram_counts = bigram_count.transform(reviews)

#review_tfidf = tfidf_vect.transform(reviews)
#print (review_tfidf.shape)
#print (review_tfidf.shape)

lda_unigram = LatentDirichletAllocation(n_components=100, random_state=0)
lda_bigram = LatentDirichletAllocation(n_components=100, random_state=1)
#lda_unigram = LdaModel(unigram_counts, num_topics=100)
#lda_bigram = LdaModel(bigram_counts, num_topics=100)

lda_unigram.fit(unigram_counts)
lda_bigram.fit(bigram_counts)

#print (lda.transform(review_tfidf[-2:]))
#train_tfidf = tfidf_vect.transform(train_x)

train_unigrams = unigram_count.transform(train_x)
train_bigrams = bigram_count.transform(train_x)

train_uni_lda = lda_unigram.transform(train_unigrams)
train_bi_lda = lda_bigram.transform(train_bigrams)

#train_uni_lda = lda_unigram[train_unigrams]
#train_bi_lda = lda_bigram[train_bigrams]

#review_lda = lda.transform(train_tfidf)

# Scale the lda between 0 and 1


print ("fitting logistic reg")

MAX_ITERATIONS = 10000

logreg = LogisticRegression(solver='lbfgs', max_iter=MAX_ITERATIONS, verbose=1)

# We need to now concatenate these
train_vec = np.concatenate((train_uni_lda, train_bi_lda), axis=1)
print (train_vec.shape)
# Scale features
#train_vec = min_max_scaler.fit_transform(train_vec)
print (train_vec)
logreg.fit(train_vec, train_y)

# Make the valid stuff
valid_unigrams = unigram_count.transform(valid_x)
valid_bigrams = bigram_count.transform(valid_x)

valid_uni_lda = lda_unigram.transform(valid_unigrams)
valid_bi_lda = lda_bigram.transform(valid_bigrams)

#valid_uni_lda = lda_unigram[valid_unigrams]
#valid_bi_lda = lda_bigram[valid_bigrams]

#valid_tfidf = tfidf_vect.transform(valid_x)
#valid_lda = lda.transform(valid_tfidf)
# Scale between 0 and 1
#valid_lda = min_max_scaler.fit_transform(valid_lda)

# Concat
valid_vec = np.concatenate((valid_uni_lda, valid_bi_lda), axis=1)
#valid_vec = min_max_scaler.fit_transform(valid_vec)
y_pred = logreg.predict(valid_vec)

y_pred_train = logreg.predict(train_vec)

log_reg_accuracy = np.mean(y_pred == valid_y)
print(log_reg_accuracy)

log_reg_accuracy_train = np.mean(y_pred_train == train_y)
print(log_reg_accuracy_train)


