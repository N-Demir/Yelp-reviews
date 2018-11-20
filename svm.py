from sklearn.svm import SVC
import collections
import operator
import numpy as np

import util

LENGTH_OF_FEATURE_VECTOR = 2_000

def get_top_words(messages, n):
    histogram = {}

    for i in range(len(messages)):
        words = get_words(messages[i])

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
    train_messages, train_labels = util.load_review_dataset('data/op_spam_v1.4/positive_polarity', ['1', '2', '3', '4'])
    test_messages, test_labels = util.load_review_dataset('data/op_spam_v1.4/positive_polarity', ['5'])

    top_words = get_top_words(train_messages, n=LENGTH_OF_FEATURE_VECTOR)

    training_features = get_features(train_messages, top_words)
    test_features = get_features(test_messages, top_words)

    svm = SVC(gamma = 'scale')
    svm.fit(training_features, train_labels)
    # y_pred = logreg.predict(test_features)
    print('Accuracy of SVM with RBF kernel on test set: {:.4f}'.format(svm.score(test_features, test_labels)))

    linearSVM = SVC(kernel = "linear")
    linearSVM.fit(training_features, train_labels)
    print('Accuracy of SVM with linear Kernel on test set: {:.4f}'.format(linearSVM.score(test_features, test_labels)))
    # np.savetxt('./outputs/logistic_regression_predictions', logistic_regression_predictions)

    # logistic_regression_accuracy = np.mean(logistic_regression_predictions == test_labels)

    # print('Logistic Regression had an accuracy of {} on the testing set'.format(logistic_regression_accuracy))


if __name__ == "__main__":
    main()
