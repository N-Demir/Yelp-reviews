import spacy
import collections
import operator
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from behavioral_analysis import getReviewerStats
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import util

spacy_en = spacy.load('en')

NUM_KFOLD_SPLITS = 5
MAX_ITERATIONS = 1000

def get_behavior_features(reviewerIDs, reviewerStats):
    fullFeatures = []
    for i, reviewerID in enumerate(reviewerIDs):
        fullFeatures.append(reviewerStats[reviewerID])
    return np.array(fullFeatures)

def main():
    if len(sys.argv) >= 2:
        dataset = sys.argv[1]
    else:
        dataset = "data/YelpChi/"

    kf = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True)

    print('Loading in dataset from {}'.format(dataset))

    # reviews, labels = util.load_yelp_dataset_full("data/YelpChi/")
    # reviews, labels = util.load_review_dataset_full('data/op_spam_v1.4')
    reviews, labels, reviewerIDs, dates, productIDs, ratings = util.load_behavioral_tsv_dataset('../data/YelpChi/labeled_behavioral_reviews.tsv')

    print('Beginning Random Forest training')

    i = 0
    train_errors = []
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

        #vectorizer = CountVectorizer()
        #vectorizer.fit(train_reviews)

        #training_word_features = vectorizer.transform(train_reviews).todense()#get_features(train_reviews, top_words)
        training_features = get_behavior_features(train_reviewerIDs, train_reviewerStats)
        #test_word_features = vectorizer.transform(test_reviews).todense()#get_features(test_reviews, top_words)
        test_features = get_behavior_features(test_reviewerIDs, reviewer_stats)
        #training_features = np.concatenate([training_word_features, training_behavior_features], axis = 1)
        #test_features = np.concatenate([test_word_features, test_behavior_features], axis = 1)

        #logreg = LogisticRegression(solver='lbfgs', max_iter=MAX_ITERATIONS, verbose = True)
        randomForest = RandomForestClassifier(n_estimators = 100, max_depth = 2)
        randomForest.fit(training_features, train_labels)
        y_pred = randomForest.predict(test_features)

        # train_errors.append(logreg.score(training_features, train_labels))
        # accuracies.append(logreg.score(test_features, test_labels))

        rand_forest_accuracy = np.mean(y_pred == test_labels)
        precision, recall, f_score = util.precision_recall_fscore(test_labels, y_pred)

        print('Logistic Regression had an accuracy of {} on the testing set for kfold {}'.format(rand_forest_accuracy, i))
        print("Precision {} Recall {} F_score {}".format(precision, recall, f_score))

        accuracies.append(rand_forest_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
        i += 1

    print('Overall, Logistic Regression had an accuracy of {} and precision {} and recall {} and f_score {}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f_scores)))


if __name__ == "__main__":
    main()
