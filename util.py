import csv

import os, fnmatch
import matplotlib.pyplot as plt
import numpy as np
import json

def load_spam_dataset(tsv_path):
    """Load the spam dataset from a TSV file

    Args:
         csv_path: Path to TSV file containing dataset.

    Returns:
        messages: A list of string values containing the text of each message.
        labels: The binary labels (0 or 1) for each message. A 1 indicates spam.
    """

    messages = []
    labels = []

    with open(tsv_path, 'r', newline='', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        for label, message in reader:
            messages.append(message)
            labels.append(1 if label == 'spam' else 0)

    return messages, np.array(labels)


def load_review_dataset(folder, folds):
	"""Load the review dataset from a given polarity folder

	Uses the Cornell dataset

	Arg:
		folder: Path to the folder that we want to crawl through
		folds: a list of the folds that we should crawl through to get the reviews

	Returns:
		reviews: A list of string values containing the text of each review
		labels: The binary labels (0 or 1) for each review. 1 indicats fake!
	"""

	reviews = []
	labels = []

	# Now we want to crawl through the folders 
	# - deceptive..
	# - truthful..
	fake = False
	for root, dirs, files in os.walk(folder):
		directory = os.path.basename(root)

		if (directory[-1: ] in folds):
			for file in files:
				with open(os.path.join(root, file)) as f:
					for review in f.readlines():
						reviews.append(review.strip())
						labels.append(1 if fake else 0)
		elif ("deceptive" in directory):
			fake = True
		elif ("truthful" in directory):
			fake = False

	return reviews, np.array(labels)
		

def write_json(filename, value):
    """Write the provided value as JSON to the given filename"""
    with open(filename, 'w') as f:
        json.dump(value, f)
     
#reviews, labels = load_review_dataset('../op_spam_v1.4/positive_polarity/', ['1'])
#print (reviews)
	