import numpy as np
import csv

path = '../data/YelpChi/labeled_reviews.tsv'

def main():
    true_reviews = []
    fake_reviews = []
    print("Parsing through file")
    with open(path) as tsvfile:
        reader = csv.reader(tsvfile, dialect='excel-tab')
        for row in reader:
            if row[1] == '0':
                true_reviews.append(row)
            else:
                fake_reviews.append(row)
    print("Creating new true review list")
    num_fake = len(fake_reviews)
    num_true = len(true_reviews)
    true_reviews_indices = np.random.choice(num_true, num_fake, replace=False)
    true_reviews_subset = []
    for index in true_reviews_indices:
        true_reviews_subset.append(true_reviews[index])
    under_sample_reviews = np.concatenate([true_reviews_subset, fake_reviews])

    print("Saving true reviews")
    with open(path[:-4]+'_balanced.tsv', 'w', newline='\n') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for row in under_sample_reviews:
            writer.writerow(row)

if __name__ == '__main__':
    main()
