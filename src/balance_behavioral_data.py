import numpy as np
import sys
import csv

meta_path_chi = '../data/YelpChi/output_meta_yelpResData_NRYRcleaned.txt'
review_path_chi = '../data/YelpChi/output_review_yelpResData_NRYRcleaned.txt'
pathMetaNYC = '../data/YelpNYC/metadata'
pathReviewNYC = '../data/YelpNYC/reviewContent'
outpath = '../data/large_behavior_balanced.tsv'

'''
Writes a file to outfile containing data from Chicago and NYC file. It includes the necessary information
to calculate the 5 behavioral statistics
'''

def main():
    true_reviews = []
    fake_reviews = []
    if len(sys.argv) == 1 or sys.argv[1]=="chicago" or sys.argv[1] == 'both':
        if len(sys.argv) >1 and sys.argv[1] == "chicago":
            outpath = "../data/YelpChi/labeled_behavioral_reviews.tsv"
        print("Parsing through Chicago file")
        with open(meta_path_chi, 'r') as meta_file, open(review_path_chi, 'r') as review_file:
            review_lines = review_file.read().splitlines()
            meta_reader = csv.reader(meta_file, delimiter = ' ')
            for review, meta in zip(review_lines, meta_reader):
                date, reviewID, reviewerID, productID, label, _, _, _, rating = meta
                label = '1' if label == 'Y' else '0'
                if label == '1':
                    fake_reviews.append([review, label, reviewerID, date, productID, rating])
                else:
                    true_reviews.append([review, label, reviewerID, date, productID, rating])
    if len(sys.argv) == 1  or sys.argv[1] == "nyc" or sys.argv[1] == "both":
        print("Parsing through NYC file")
        if len(sys.argv) >1 and sys.argv[1] == "nyc":
            outpath = "../data/YelpNYC/labeled_behavioral_reviews.tsv"
        with open(pathMetaNYC) as metaNYC, open(pathReviewNYC) as reviewNYC:
            meta_reader = csv.reader(metaNYC, dialect = 'excel-tab')
            review_reader = csv.reader(reviewNYC, dialect = 'excel-tab')
            for metarow, reviewrow in zip(meta_reader, review_reader):
                reviewerID, productID, rating, label, date = metarow
                label = '0' if label == '1' else '1'
                if label == '0':
                    true_reviews.append([reviewrow[3], label, reviewerID, date, productID, rating])
                else:
                    fake_reviews.append([reviewrow[3], label, reviewerID, date, productID, rating])
    else:
        print("Please input 'both', 'chicago', or 'nyc' as an argument ")

    print("Creating new true review list")
    num_fake = len(fake_reviews)
    num_true = len(true_reviews)
    print("numfake: ", num_fake)
    print("numtrue: ", num_true)

    true_reviews_indices = np.random.choice(num_true, num_fake, replace=False)
    true_reviews_subset = []
    for index in true_reviews_indices:
        true_reviews_subset.append(true_reviews[index])
    under_sample_reviews = np.concatenate([true_reviews_subset, fake_reviews])

    print("Saving true reviews")
    with open(outpath, 'w', newline='\n') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for row in under_sample_reviews:
            writer.writerow(row)

if __name__ == '__main__':
    main()
