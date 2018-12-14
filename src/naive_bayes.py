import spacy
import collections

import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path
from datetime import datetime
from behavioral_analysis import getReviewerStats
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

import sys
import util
import csv

NUM_KFOLD_SPLITS = 5
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
REPOSITORY_NAME = 'Yelp-reviews'

spacy_en = spacy.load('en')
path = '../data/large_balanced.tsv'

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


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    index_dict = collections.defaultdict(int)
    message_counts = collections.defaultdict(int)

    index = 0
    for message in messages:
        unique_words = set(get_words(message))

        for word in unique_words:
            message_counts[word] += 1

            # We have seen it for the fifth time so let us add it to
            # the index_dict
            if (message_counts[word] == 5):
                index_dict[word] = index
                index += 1

    return index_dict

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    mat = np.zeros((len(messages), len(word_dictionary)))

    for i in range(len(messages)):
        words = get_words(messages[i])

        for word in words:
            if word in word_dictionary:
                word_indx = word_dictionary[word]
                mat[i][word_indx] += 1

    return mat
    # *** END CODE HERE ***

def analyze_results(test_reviews, true_labels, predictions):
    '''
    Compare the predictions to the true labels and see where we go wrong

    Args:
        test_reviews: The actual reviews that we analys at test time
        true_labels: The ground truth labeling
        predictions: Our models predictions of the labels
    '''
    num_incorrect = 0
    for i in range(true_labels.shape[0]):
        # When the prediction is wrong what does the thing look like
        if (true_labels[i] != int(predictions[i])):
            num_incorrect += 1
            print ("Incorrect Prediction #: %d" % (num_incorrect))
            if (true_labels[i] == 0):
                print ("Should have prediced Real Review")
            else:
                print ("Should have prediced Fake Review")

            print("Review:")
            print (test_reviews[i])

            print ("\n\n")

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
    NB_folder = checkpoints_folder / 'NaiveBayes'
    cur_folder = NB_folder / CURRENT_TIME

    checkpoints_folder.mkdir(exist_ok=True)
    NB_folder.mkdir(exist_ok=True)
    cur_folder.mkdir(exist_ok=True)

    return cur_folder

def saveMetrics(prefix, accuracy, precisions, recalls, f1_scores, epoch):
    real_precision, fake_precision = precisions
    real_recall, fake_recall = recalls
    real_f1_score, fake_f1_score = f1_scores

    path = setupCheckpoints()

    accuracy_path = path / '{}-accuracy.txt'.format(prefix)

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
    writeMetric(real_precision_path, real_precision)
    writeMetric(fake_precision_path, fake_precision)
    writeMetric(real_recall_path, real_recall)
    writeMetric(fake_recall_path, fake_recall)
    writeMetric(real_f1_score_path, real_f1_score)
    writeMetric(fake_f1_score_path, fake_f1_score)

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

    print('Loading in dataset from {}'.format(path))

    # reviews, labels = util.load_yelp_dataset_full(dataset)
    # reviews, labels = util.load_review_dataset_full('data/op_spam_v1.4')
    # reviews, labels = util.load_tsv_dataset(path)
    reviews, labels, reviewerIDs, dates, productIDs, ratings = util.load_behavioral_tsv_dataset('../data/YelpChi/labeled_behavioral_reviews.tsv')

    print('Beginning Naive Bayes training')

    i = 0
    accuracies = []
    real_precisions = []
    fake_precisions = []
    real_recalls = []
    fake_recalls = []
    real_f_scores = []
    fake_f_scores = []
    reviewer_stats = getReviewerStats(reviews, labels, reviewerIDs, dates, productIDs, ratings)
    confusionMatrix = np.zeros((2, 2))
    for train_index, test_index in kf.split(reviews):
        train_reviews, train_labels = reviews[train_index], labels[train_index]
        test_reviews, test_labels = reviews[test_index], labels[test_index]

        train_reviewerIDs, train_dates, train_productIDs, train_ratings = reviewerIDs[train_index], dates[train_index], productIDs[train_index], ratings[train_index]
        test_reviewerIDs, test_dates, test_productIDs, test_ratings = reviewerIDs[test_index], dates[test_index], productIDs[test_index], ratings[test_index]

        train_reviewerStats = getReviewerStats(train_reviews, train_labels, train_reviewerIDs, train_dates, train_productIDs, train_ratings)
        training_behavior_features = get_behavior_features(train_reviewerIDs, train_reviewerStats)
        test_behavior_features = get_behavior_features(test_reviewerIDs, reviewer_stats)


        dictionary = create_dictionary(train_reviews)
        train_reviews = np.concatenate([transform_text(train_reviews, dictionary), training_behavior_features], axis = 1)
        test_reviews = np.concatenate([transform_text(test_reviews, dictionary), test_behavior_features], axis = 1)

        clf = MultinomialNB()
        clf.fit(train_reviews, train_labels)

        naive_bayes_predictions = clf.predict(test_reviews)
        confusionMatrix += confusion_matrix(test_labels, naive_bayes_predictions)

        # analyze_results(test_reviews, test_labels, naive_bayes_predictions)

        naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)
        precisions, recalls, f_scores = util.precision_recall_fscore(test_labels, naive_bayes_predictions)
        real_precision, fake_precision = precisions
        real_recall, fake_recall = recalls
        real_f_score, fake_f_score = f_scores

        print('Naive Bayes had an accuracy of {} on the testing set for kfold {}'.format(naive_bayes_accuracy, i))
        print("Precisions {} Recalls {} F_scores {}".format(precisions, recalls, f_scores))

        accuracies.append(naive_bayes_accuracy)
        real_precisions.append(real_precision)
        fake_precisions.append(fake_precision)
        real_recalls.append(real_recall)
        fake_recalls.append(fake_recall)
        real_f_scores.append(real_f_score)
        fake_f_scores.append(fake_f_score)
        i += 1

    avg_accuracy = np.mean(accuracies)

    print('Overall, Naive Bayes had an accuracy of {}.'.format(avg_accuracy))
    print('Real reviews with precision {} and recall {} and f_score {}'.format(np.mean(real_precisions), np.mean(real_recalls), np.mean(real_f_scores)))
    print('Fake reviews with precision {} and recall {} and f_score {}'.format(np.mean(fake_precisions), np.mean(fake_recalls), np.mean(fake_f_scores)))
    print(confusionMatrix)
    # saveMetrics('valid', avg_accuracy, [np.mean(real_precisions), np.mean(fake_precisions)], [np.mean(real_recalls), np.mean(fake_recalls)], [np.mean(real_f_scores), np.mean(fake_f_scores)], i)

if __name__ == "__main__":
    main()
