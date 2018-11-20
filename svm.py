from sklearn.svm import SVC
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

NUM_KFOLD_SPLITS = 20

def main():
    kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)

    reviews, labels = util.load_review_dataset_full('data/op_spam_v1.4')

    RBFaccuracies = []
    linearAccuracies = []
    for train_index, test_index in kf.split(reviews):
        train_reviews, train_labels = reviews[train_index], labels[train_index]
        test_reviews, test_labels = reviews[test_index], labels[test_index]

        top_words = get_top_words(train_reviews, n=LENGTH_OF_FEATURE_VECTOR)

        training_features = get_features(train_reviews, top_words)
        test_features = get_features(test_reviews, top_words)

        svm = SVC(gamma = 'scale')
        svm.fit(training_features, train_labels)
        RBFaccuracies.append(svm.score(test_features, test_labels))

        linearSVM = SVC(kernel = "linear")
        linearSVM.fit(training_features, train_labels)
        linearAccuracies.append(linearSVM.score(test_features, test_labels))

    print('Accuracy of SVM with RBF Kernel on test set: {:.3f}'.format(np.mean(RBFaccuracies)))
    print('Accuracy of SVM with linear Kernel on test set: {:.3f}'.format(np.mean(linearAccuracies)))
    # np.savetxt('./outputs/logistic_regression_predictions', logistic_regression_predictions)

    # logistic_regression_accuracy = np.mean(logistic_regression_predictions == test_labels)

    # print('Logistic Regression had an accuracy of {} on the testing set'.format(logistic_regression_accuracy))


if __name__ == "__main__":
    main()
