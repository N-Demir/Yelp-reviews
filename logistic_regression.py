import collections
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import numpy as np

import util

NUM_KFOLD_SPLITS = 5
MAX_ITERATIONS = 1000
LENGTH_OF_FEATURE_VECTOR = 2_000

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

    return [word.lower() for word in message.split(" ")]

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

    # print (len(index_dict))
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

def get_top_words(messages, n):
    histogram = {}

    for i in range(len(messages)):
        words = get_words(messages[i])

        for word in words:
            histogram[word] = histogram.get(word, 0) + 1

    sorted_by_value = sorted(histogram.items(), key=operator.itemgetter(1), reverse=True)

    return dict((tup[0], i) for i, tup in enumerate(sorted_by_value[:n]))

def get_features(messages, top_words):
    features = []

    for message in messages:
        feature = np.zeros(LENGTH_OF_FEATURE_VECTOR)

        words = get_words(message)
        for word in words:
            if word in top_words:
                feature[top_words[word]] += 1

        features.append(feature)

    return features




def main():
    kf = KFold(n_splits=NUM_KFOLD_SPLITS)

    messages, labels = util.load_review_dataset_full('data/op_spam_v1.4/positive_polarity')

    accuracies = []
    for train_index, test_index in kf.split(messages):
        train_messages, train_labels = messages[train_index], labels[train_index]
        test_messages, test_labels = messages[test_index], labels[test_index]

        top_words = get_top_words(train_messages, n=LENGTH_OF_FEATURE_VECTOR)

        training_features = get_features(train_messages, top_words)
        test_features = get_features(test_messages, top_words)

        logreg = LogisticRegression(solver='lbfgs', max_iter=MAX_ITERATIONS)
        logreg.fit(training_features, train_labels)
        y_pred = logreg.predict(test_features)

        accuracies.append(logreg.score(test_features, test_labels))
        
    print('Accuracy of logistic regression classifier on test set: {:.3f}'.format(np.mean(accuracies)))


if __name__ == "__main__":
    main()