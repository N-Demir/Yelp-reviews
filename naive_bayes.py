import collections

import numpy as np
from sklearn.model_selection import KFold

import util

NUM_KFOLD_SPLITS = 20

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


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # For the model we want:
    #   for each word have a count of the 
    #   number of times it appears in spam and not spam
    #
    #   number of words in spam and not spam emails

    word_count_not_fake = np.zeros((matrix.shape[1]))
    word_count_fake = np.zeros((matrix.shape[1]))

    count_words_fake = 0
    count_words_not_fake = 0

    count_fake_reviews = 0

    for i in range(matrix.shape[0]):
        message_label = labels[i]
        if (message_label == 1):
            count_fake_reviews += 1

        for j in range(matrix.shape[1]):
            count_word = matrix[i][j]

            if (message_label == 1):
                count_words_fake += count_word
                word_count_fake[j] += count_word
            else:
                count_words_not_fake += count_word
                word_count_not_fake[j] += count_word

    prob_fake = 1.0 * count_fake_reviews / matrix.shape[0]
    return (word_count_not_fake, word_count_fake, count_words_fake, count_words_not_fake, prob_fake)



def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    word_count_not_fake, word_count_fake, count_words_fake, count_words_not_fake, prob_fake = model

    vocab_size = len(word_count_not_fake)

    predictions = np.zeros((matrix.shape[0]))

    for i in range(matrix.shape[0]):
        log_prob_fake = np.log(prob_fake)
        log_prob_not_fake = np.log(1 - prob_fake)

        for j in range(matrix.shape[1]):
            count_word = matrix[i][j]
            if (count_word != 0):
                log_prob_fake += count_word * np.log((1.0 * word_count_fake[j] + 1) / (count_words_fake + vocab_size))
                log_prob_not_fake += count_word * np.log((1.0 * word_count_not_fake[j] + 1 )/ (count_words_not_fake + vocab_size))

        if (log_prob_fake > log_prob_not_fake):
            predictions[i] = 1
        else:
            predictions[i] = 0

    return predictions




def get_top_n_naive_bayes_words(model, dictionary, n):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids
        n: The number of top words that we want to look at

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    word_count_not_fake, word_count_fake, count_words_fake, count_words_not_fake, prob_fake = model
    prob_for_word = []
    vocab_size = len(word_count_not_fake)

    for word, indx in dictionary.items():
        # Calc p(word | spam)
        prob_fake = np.log((1.0 * word_count_fake[indx] + 1) / (count_words_fake + vocab_size))
        # Calc p(word | not_spam)
        prob_not_spam = np.log((1.0 * word_count_not_fake[indx] + 1) / (count_words_not_fake + vocab_size))

        ratio = prob_fake - prob_not_spam

        prob_for_word.append((word, ratio))

    # Sort based on ratio
    prob_for_word = sorted(prob_for_word, key=lambda x: x[1], reverse=True)

    top_words = []
    for i in range(n):
        top_words.append(prob_for_word[i][0])

    return top_words


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


'''
def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    best_accuracy = -1.
    best_radius = -1.
    for radius in radius_to_consider:
        prediction_for_radius = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(prediction_for_radius == val_labels)

        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            best_radius = radius

    return best_radius
    # *** END CODE HERE ***
'''

def main():
    kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)

    reviews, labels = util.load_yelp_dataset_full("data/YelpChi/")
    # reviews, labels = util.load_review_dataset_full('data/op_spam_v1.4')

    accuracies = []
    for train_index, test_index in kf.split(reviews):
        train_reviews, train_labels = reviews[train_index], labels[train_index]
        test_reviews, test_labels = reviews[test_index], labels[test_index]
    
        dictionary = create_dictionary(train_reviews)

        util.write_json('./outputs/dictionary', dictionary)

        train_matrix = transform_text(train_reviews, dictionary)
        test_matrix = transform_text(test_reviews, dictionary)

        naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

        naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

        np.savetxt('./outputs/naive_bayes_predictions', naive_bayes_predictions)

        naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

        print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

        analyze_results(test_reviews, test_labels, naive_bayes_predictions)

        top_n_words = get_top_n_naive_bayes_words(naive_bayes_model, dictionary, 10)

        print('The top 5 indicative words for Naive Bayes are: ', top_n_words)

        accuracies.append(naive_bayes_accuracy)

        precision, recall, f_score = util.precision_recall_fscore(test_labels, naive_bayes_predictions)
        print("Precision {} Recall {} F_score {}".format(precision, recall, f_score))
    print('Overall, Naive Bayes had an accuracy of: ', np.mean(accuracies))

    # precision, recall, f_score = util.precision_recall_fscore(test_labels, naive_bayes_predictions)
    
    '''
    util.write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))
    '''


if __name__ == "__main__":
    main()
