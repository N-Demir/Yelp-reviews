'''
	Utility method to compare prediction.txt files of two different models
	
	Usage: python computeOutputSimilarity.py <pred_model_1.txt> <pred_model_2.txt>
	
	Txt file format: <epoch>,<value>
'''

import sys

path_1 = sys.argv[1]
path_2 = sys.argv[2]

total = 0.
num = 0.

with open(path_1, 'r') as f_1:
	with open(path_2, 'r') as f_2:
	    lines_1 = f_1.readlines()
	    lines_2 = f_2.readlines()

	    for i in range(len((lines_1))):
	    	if lines_1[i] == lines_2[i]:
	    		total += 1
	    	num += 1.0

print('Got similarity: {}'.format(total / num))