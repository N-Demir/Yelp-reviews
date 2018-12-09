#from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
import collections
import operator
import numpy as np

import util

LENGTH_OF_FEATURE_VECTOR = 2_000

def get_top_words(reviews, n):
    histogram = {}

    for i in range(len(reviews)):
        words = get_words(reviews[i])

        for word in words:
            histogram[word] = histogram.get(word, 0) + 1

    sorted_by_value = sorted(histogram.items(), key=operator.itemgetter(1), reverse=True)

    return dict((tup[0], i) for i, tup in enumerate(sorted_by_value[:n]))

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

    return index_dict

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
    SVM_folder = checkpoints_folder / 'SVM'
    cur_folder = SVM_folder / CURRENT_TIME
    
    checkpoints_folder.mkdir(exist_ok=True)
    SVM_folder.mkdir(exist_ok=True)
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

NUM_KFOLD_SPLITS = 10

def main():
    kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)

    reviews, labels = util.load_yelp_dataset_full("data/Yelp-Data/")

    # reviews, labels = util.load_review_dataset_full('data/op_spam_v1.4')

    i = 0
    accuracies = []
    real_precisions = []
    fake_precisions = []
    real_recalls = []
    fake_recalls = []
    real_f_scores = []
    fake_f_scores = []
    for train_index, test_index in kf.split(reviews):
        print("Beginning train index ", i)
        train_reviews, train_labels = reviews[train_index], labels[train_index]
        test_reviews, test_labels = reviews[test_index], labels[test_index]

        top_words = get_top_words(train_reviews, n=LENGTH_OF_FEATURE_VECTOR)

        dictionary = create_dictionary(train_reviews)
        train_reviews = transform_text(train_reviews, dictionary)
        test_reviews = transform_text(test_reviews, dictionary)

        linearSVM = SGDClassifier()
        linearSVM.fit(train_reviews, train_labels)
        SVM_predictions = linearSVM.predict(test_reviews)

        SVM_accuracy = np.mean(SVM_predictions == test_labels)

        precisions, recalls, f_scores = util.precision_recall_fscore(test_labels, SVM_predictions)
        real_precision, fake_precision = precisions
        real_recall, fake_recall = recalls
        real_f_score, fake_f_score = f_scores

        accuracies.append(SVM_accuracy)
        real_precisions.append(real_precision)
        fake_precisions.append(fake_precision)
        real_recalls.append(real_recall)
        fake_recalls.append(fake_recall)
        real_f_scores.append(real_f_score)
        fake_f_scores.append(fake_f_score)

        i += 1

    avg_accuracy = np.mean(accuracies)

    print('Overall, SVM had an accuracy of {}.'.format(avg_accuracy))
    print('Real reviews with precision {} and recall {} and f_score {}'.format(np.mean(real_precisions), np.mean(real_recalls), np.mean(real_f_scores)))
    print('Fake reviews with precision {} and recall {} and f_score {}'.format(np.mean(fake_precisions), np.mean(fake_recalls), np.mean(fake_f_scores)))

    saveMetrics('valid', avg_accuracy, [np.mean(real_precisions), np.mean(fake_precisions)], [np.mean(real_recalls), p.mean(fake_recalls)], [np.mean(real_f_scores), np.mean(fake_f_scores)], i)

if __name__ == "__main__":
    main()
