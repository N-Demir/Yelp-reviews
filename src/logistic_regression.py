import spacy
import collections
import operator
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from behavioral_analysis import getReviewerStats
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import util

spacy_en = spacy.load('en')

NUM_KFOLD_SPLITS = 5
MAX_ITERATIONS = 1000
LENGTH_OF_FEATURE_VECTOR = 1000

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """
    return [tok.text for tok in spacy_en.tokenizer(str(message))]

def create_dictionary(reviews):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training reviews. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five reviews.

    Args:
        reviews: A list of strings containing SMS reviews

    Returns:
        A python dict mapping words to integers.
    """

    index_dict = collections.defaultdict(int)
    message_counts = collections.defaultdict(int)

    index = 0
    for message in reviews:
        unique_words = set(get_words(message))

        for word in unique_words:
            message_counts[word] += 1

            # We have seen it for the fifth time so let us add it to
            # the index_dict
            if (message_counts[word] == 5):
                index_dict[word] = index
                index += 1

    # print (len(index_dict))
    return index_dict



def transform_text(reviews, word_dictionary):
    """Transform a list of text reviews into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        reviews: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***


    mat = np.zeros((len(reviews), len(word_dictionary)))

    for i in range(len(reviews)):
        words = get_words(reviews[i])

        for word in words:
            if word in word_dictionary:
                word_indx = word_dictionary[word]
                mat[i][word_indx] += 1

    return mat
    # *** END CODE HERE ***

def get_top_words(reviews, n):
    histogram = {}

    for i in range(len(reviews)):
        words = get_words(reviews[i])

        for word in words:
            histogram[word] = histogram.get(word, 0) + 1

    sorted_by_value = sorted(histogram.items(), key=operator.itemgetter(1), reverse=True)

    return dict((tup[0], i) for i, tup in enumerate(sorted_by_value[:n]))

def get_features(reviews, top_words):
    features = []

    for message in reviews:
        feature = np.zeros(LENGTH_OF_FEATURE_VECTOR)

        words = get_words(message)
        for word in words:
            if word in top_words:
                feature[top_words[word]] += 1

        features.append(feature)

    return features

def get_behavior_features(reviewerIDs, reviewerStats):
    fullFeatures = []
    for i, reviewerID in enumerate(reviewerIDs):
        fullFeatures.append(reviewerStats[reviewerID])
    return np.array(fullFeatures)

def main():
    # if len(sys.argv) >= 2:
    #     dataset = sys.argv[1]
    # else:
    #     dataset = "data/YelpChi/"

    kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)

    # print('Loading in dataset from {}'.format(dataset))

    # reviews, labels = util.load_yelp_dataset_full("data/YelpChi/")
    # reviews, labels = util.load_review_dataset_full('data/op_spam_v1.4')
    reviews, labels, reviewerIDs, dates, productIDs, ratings = util.load_behavioral_tsv_dataset('../data/YelpChi/labeled_behavioral_reviews.tsv')

    print('Beginning Logistic Regression training')

    i = 0
    train_accuracy = []
    accuracies = []
    precisions = []
    recalls = []
    f_scores = []
    reviewer_stats = getReviewerStats(reviews, labels, reviewerIDs, dates, productIDs, ratings)
    for train_index, test_index in kf.split(reviews):
        train_reviews, train_labels = reviews[train_index], labels[train_index]
        test_reviews, test_labels = reviews[test_index], labels[test_index]
        # behavioral data
        train_reviewerIDs, train_dates, train_productIDs, train_ratings = reviewerIDs[train_index], dates[train_index], productIDs[train_index], ratings[train_index]
        test_reviewerIDs, test_dates, test_productIDs, test_ratings = reviewerIDs[test_index], dates[test_index], productIDs[test_index], ratings[test_index]

        train_reviewerStats = getReviewerStats(train_reviews, train_labels, train_reviewerIDs, train_dates, train_productIDs, train_ratings)

        # top_words = get_top_words(train_reviews, n=LENGTH_OF_FEATURE_VECTOR)

        vectorizer = CountVectorizer()
        vectorizer.fit(train_reviews)

        training_word_features = vectorizer.transform(train_reviews).todense()#get_features(train_reviews, top_words)
        training_behavior_features = get_behavior_features(train_reviewerIDs, train_reviewerStats)
        test_word_features = vectorizer.transform(test_reviews).todense()#get_features(test_reviews, top_words)
        test_behavior_features = get_behavior_features(test_reviewerIDs, reviewer_stats)
        training_features = np.concatenate([training_behavior_features], axis = 1)
        test_features = np.concatenate([test_behavior_features], axis = 1)
        training_features = normalize(training_features)
        test_features = normalize(test_features)

        logreg = LogisticRegression(solver='lbfgs', max_iter=MAX_ITERATIONS, verbose = True)
        logreg.fit(training_features, train_labels)
        y_pred = logreg.predict(test_features)

        train_accuracy.append(logreg.score(training_features, train_labels))
        # accuracies.append(logreg.score(test_features, test_labels))

        log_reg_accuracy = np.mean(y_pred == test_labels)
        precision, recall, f_score = util.precision_recall_fscore(test_labels, y_pred)

        print('Logistic Regression had an accuracy of {} on the testing set for kfold {}'.format(log_reg_accuracy, i))
        print("Precision {} Recall {} F_score {}".format(precision, recall, f_score))

        print('Double checking {}'.format(logreg.score(test_features, test_labels)))

        accuracies.append(log_reg_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
        i += 1

    print('Overall, Logistic Regression had an accuracy of {} and precision {} and recall {} and f_score {}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f_scores)))
    print('Train accuracy {}'.format(np.mean(train_accuracy)))

if __name__ == "__main__":
    main()
