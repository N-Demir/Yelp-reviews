import numpy as np
import csv

path = '../data/YelpChi/labeled_reviews.tsv'
pathMetaNYC = '../data/YelpNYC/metadata'
pathReviewNYC = '../data/YelpNYC/reviewContent'
outpath = '../data/large_balanced.tsv'

def main():
    true_reviews = []
    fake_reviews = []
    print("Parsing through Chicago file")
    with open(path) as tsvfile:
        reader = csv.reader(tsvfile, dialect='excel-tab')
        for row in reader:
            if row[1] == '0':
                true_reviews.append(row)
            else:
                fake_reviews.append(row)
    print("Parsing through NYC file")
    with open(pathMetaNYC) as metaNYC, open(pathReviewNYC) as reviewNYC:
        meta_reader = csv.reader(metaNYC, dialect = 'excel-tab')
        review_reader = csv.reader(reviewNYC, dialect = 'excel-tab')
        for metarow, reviewrow in zip(meta_reader, review_reader):
            if metarow[3] == '1':
                true_reviews.append([reviewrow[3], '0'])
            else:
                fake_reviews.append([reviewrow[3], '1'])

    print("Creating new true review list")
    num_fake = len(fake_reviews)
    num_true = len(true_reviews)
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
