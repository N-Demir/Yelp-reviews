import spacy
import collections

import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

import sys
import util
import csv

NUM_KFOLD_SPLITS = 20

spacy_en = spacy.load('en')
path = '../data/YelpChi/labeled_reviews_balanced.tsv'

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

def main():
    if len(sys.argv) >= 2:
        dataset = sys.argv[1]
    else:
        dataset = "data/YelpChi/"

    kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)

    print('Loading in dataset from {}'.format(dataset))

    # reviews, labels = util.load_yelp_dataset_full(dataset)
    # reviews, labels = util.load_review_dataset_full('data/op_spam_v1.4')
    reviews, labels = util.load_tsv_dataset(path)

    print('Beginning Naive Bayes training')

    i = 0
    accuracies = []
    precisions = []
    recalls = []
    f_scores = []
    for train_index, test_index in kf.split(reviews):
        train_reviews, train_labels = reviews[train_index], labels[train_index]
        test_reviews, test_labels = reviews[test_index], labels[test_index]

        dictionary = create_dictionary(train_reviews)
        train_reviews = transform_text(train_reviews, dictionary)
        test_reviews = transform_text(test_reviews, dictionary)

        clf = MultinomialNB()
        clf.fit(train_reviews, train_labels)

        naive_bayes_predictions = clf.predict(test_reviews)

        np.savetxt('../outputs/naive_bayes_predictions', naive_bayes_predictions)

        # analyze_results(test_reviews, test_labels, naive_bayes_predictions)

        naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)
        precision, recall, f_score = util.precision_recall_fscore(test_labels, naive_bayes_predictions)

        print('Naive Bayes had an accuracy of {} on the testing set for kfold {}'.format(naive_bayes_accuracy, i))
        print("Precision {} Recall {} F_score {}".format(precision, recall, f_score))

        accuracies.append(naive_bayes_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
        i += 1

    print('Overall, Naive Bayes had an accuracy of {} and precision {} and recall {} and f_score {}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f_scores)))

if __name__ == "__main__":
    main()
