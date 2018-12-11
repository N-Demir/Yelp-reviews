import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bisect import bisect_left
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import util
import csv

meta_path = '../data/YelpChi/output_meta_yelpResData_NRYRcleaned.txt'
review_path = '../data/YelpChi/output_review_yelpResData_NRYRcleaned.txt'

def getReviewerStats(reviewsText, labels, reviewerIDs, dates, productIDs, ratings):
    reviewer_dict = defaultdict(list)
    product_ratings = defaultdict(list)

    for i in range(len(reviewsText)):
        reviewer_dict[reviewerIDs[i]].append([dates[i], productIDs[i], labels[i], ratings[i], reviewsText[i]])
        product_ratings[productIDs[i]].append(float(ratings[i]))

    avg_product_scores = {}
    for productID, scores in product_ratings.items():
        avg_product_scores[productID] = np.mean(scores)

    reviewer_stats = {}
    for reviewer, reviews in reviewer_dict.items():
        reviews = np.array(reviews)

        # Calculating maximum_number_of_reviews
        counter = Counter(reviews[:, 0])
        maximum_number_of_reviews = counter.most_common(1)[0][1]  # Tiny impact. Does this even mean anything?

        # Calculating avg review value
        reviewer_ratings = reviews[:, 3]
        reviewer_ratings = np.array(list(map(float, reviewer_ratings)))
        avg_value_of_review = np.mean(reviewer_ratings) # No impact

        # Calculating avg_review_length
        reviews_text = reviews[:, 4]
        avg_review_length = np.mean([len(sentence.split()) for sentence in reviews_text]) #5 %

        # Calculate avg_deviation
        deviation = []
        for review_info in reviews:
            deviation.append(np.mean(np.abs(product_ratings[review_info[1]]-np.array([float(review_info[3])]))))# deviation.append(np.abs(avg_product_scores[review_info[1]] - float(review_info[3])))
        avg_deviation = np.mean(deviation)

        # Calculate maimum_content_similarity
        maximum_content_similarity = 0
        if len(reviews_text) > 1:
            vectorizer = CountVectorizer()
            counts_matrix = vectorizer.fit_transform(reviews_text)
            cosine_similarities = cosine_similarity(counts_matrix)
            np.fill_diagonal(cosine_similarities, 0)
            maximum_content_similarity = np.max(cosine_similarities) # No impact either ??

        reviewer_stats[reviewer] = [avg_deviation, avg_review_length]
    return reviewer_stats

def main():
    reviewer_dict = defaultdict(list)
    product_ratings = defaultdict(list)
    print("Parsing through Chicago file")
    with open(meta_path, 'r') as meta_file, open(review_path, 'r') as review_file:
        review_lines = review_file.read().splitlines()
        meta_reader = csv.reader(meta_file, delimiter = ' ')
        for review, meta in zip(review_lines, meta_reader):
            date, reviewID, reviewerID, productID, label, _, _, _, rating = meta
            reviewer_dict[reviewerID].append([date, reviewID, productID, 1 if label == 'Y' else 0, rating, review])
            product_ratings[productID].append(int(rating))

    avg_product_scores = {}
    for productID, scores in product_ratings.items():
        avg_product_scores[productID] = np.mean(scores)

    reviewer_stats = {}
    fakeSimil = []
    realSimil = []
    fakeLength = []
    realLength = []
    fakeMax = []
    realMax = []
    fakePositive = []
    realPositive = []
    fakeDeviation = []
    realDeviation = []
    for reviewer, reviews in reviewer_dict.items():
        reviews = np.array(reviews)

        # Calculating maximum_number_of_reviews
        counter = Counter(reviews[:, 0])
        maximum_number_of_reviews = counter.most_common(1)[0][1]

        # Calculating percentage_of_positive_reviews
        ratings = reviews[:, 4]
        ratings = np.array(list(map(int, ratings)))
        percentage_of_positive_reviews = len(ratings[ratings>=4])/len(ratings)

        # Calculating avg_review_length
        reviews_text = reviews[:, 5]
        avg_review_length = np.mean([len(sentence.split()) for sentence in reviews_text])

        # Calculate avg_deviation
        deviation = []
        for review_info in reviews:
            deviation.append(np.abs(avg_product_scores[review_info[2]] - int(review_info[4])))
        avg_deviation = np.mean(deviation)

        # Calculate maimum_content_similarity
        maximum_content_similarity = 0
        if len(reviews_text) > 1:
            vectorizer = CountVectorizer()
            counts_matrix = vectorizer.fit_transform(reviews_text)
            cosine_similarities = cosine_similarity(counts_matrix)
            np.fill_diagonal(cosine_similarities, 0)
            maximum_content_similarity = np.max(cosine_similarities)
        np.array(list(map(int, reviews[:,3])))
        if np.max(np.array(list(map(int, reviews[:,3])))) == 1:
            fakeSimil.append(maximum_content_similarity)
            fakeLength.append(avg_review_length)
            fakeMax.append(maximum_number_of_reviews)
            fakePositive.append(percentage_of_positive_reviews)
            fakeDeviation.append(avg_deviation)
        else:
            realSimil.append(maximum_content_similarity)
            realLength.append(avg_review_length)
            realMax.append(maximum_number_of_reviews)
            realPositive.append(percentage_of_positive_reviews)
            realDeviation.append(avg_deviation)
        reviewer_stats[reviewer] = [maximum_number_of_reviews, percentage_of_positive_reviews, avg_review_length, avg_deviation, maximum_content_similarity]
    # print("Average Content Similarity")
    # print("fake: ", np.mean(fakeSimil))
    # print("real: ",np.mean(realSimil))
    # print("Average review length")
    # print("fake: ",np.mean(fakeLength))
    # print("real: ",np.mean(realLength))
    # print("Average max number of reviews")
    # print("fake: ",np.mean(fakeMax))
    # print("real: ", np.mean(realMax))
    # print("Average avg_deviation")
    # print("fake: ", np.mean(fakeDeviation))
    # print("real: ", np.mean(realDeviation))

def graph_cdfs():
    reviews, labels, reviewerIDs, dates, productIDs, ratings = util.load_behavioral_tsv_dataset('../data/YelpChi/labeled_behavioral_reviews.tsv')

    fakeSimil = []
    realSimil = []
    fakeLength = []
    realLength = []
    fakeMax = []
    realMax = []
    fakePositive = []
    realPositive = []
    fakeDeviation = []
    realDeviation = []
    reviewer_stats = getReviewerStats(reviews, labels, reviewerIDs, dates, productIDs, ratings)
    for reviewerID, stats in reviewer_stats.items():
        reviewer_idxs = np.where(reviewerIDs==reviewerID)
        reviewer_labels = labels[reviewer_idxs]
        if np.max(np.array(list(map(int, reviewer_labels)))) == 1:
            fakeSimil.append(stats[4])
            fakeLength.append(stats[2])
            fakeMax.append(stats[0])
            fakePositive.append(stats[1])
            fakeDeviation.append(stats[3])
        else:
            realSimil.append(stats[4])
            realLength.append(stats[2])
            realMax.append(stats[0])
            realPositive.append(stats[1])
            realDeviation.append(stats[3])

    data = [(fakeSimil, realSimil), (fakeLength, realLength), (fakeMax, realMax), (fakePositive, realPositive), (fakeDeviation, realDeviation)]
    titles = ["Maximum content Similarity", "Average content length", "Maximum number of reviews / day", "Percent of positive reviews", "Average rating deviation"]

    fig, axes = plt.subplots(1, 5)
    for i, (fake, real) in enumerate(data):
        sorted_real = np.sort(real)
        sorted_fake = np.sort(fake)

        real_yvals = np.arange(len(sorted_real))/float(len(sorted_real)-1)
        fake_yvals = np.arange(len(sorted_fake))/float(len(sorted_fake)-1)
        axes[i].plot(sorted_real, real_yvals, label = "real")
        axes[i].plot(sorted_fake, fake_yvals, 'r', label = "fake")
        axes[i].set_title(titles[i])
    axes[4].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

if __name__ == '__main__':
    #main()
    graph_cdfs()
