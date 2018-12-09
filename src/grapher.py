'''
	Utility method to make graphs for all the .txt files in a folder. Can plot
	an average over a given number of epochs.
	
	Usage: python grapher.py <folder_path> <num_to_average_over>
	
	Txt file format: <epoch>,<value>
'''

import matplotlib.pyplot as plt
import sys
import os

avg_over = float(sys.argv[2])

def walkThroughFolder(folder_name):
	for root, dirs, files in os.walk(folder_name):
		directory = os.path.basename(root)

		for file in files:
			makeGraph(root, file)

		for dr in dirs:
			walkThroughFolder(os.path.join(root, dr))


def makeGraph(root, file):
	file_path = os.path.join(root, file)
	if file_path[-4:] != '.txt': return

	title = ' '.join(file[:-4].split('-')).capitalize()

	x = []
	y = []

	with open(file_path, "r") as f:
		i = 0.
		y_s = []
		for line in f.readlines():
			split_lines = line.split(",")
			y_s.append(float(split_lines[1]))
			i += 1.
			if i % avg_over == 0:
				x.append(i)
				y.append(sum(y_s) / avg_over)
				y_s = []

	plt.plot(x, y)
	plt.title(title)
	plt.savefig(file_path[:-4] + '.png')
	plt.close()

walkThroughFolder(sys.argv[1])
