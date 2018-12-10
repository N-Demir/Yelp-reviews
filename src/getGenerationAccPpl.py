'''
    Simple script to extract Accuracy and Perplexity from generation data
	
	Usage: python getGenerationAccPpl.py <folder_name> <train_output_file_name>

	Example: python getGenerationAccPpl.py 4-gpu run_1.txt
'''

import sys
import re

from pathlib import Path

REPOSITORY_NAME = 'Yelp-reviews'

def setupCheckpoints():
    def getRepositoryPath():
        """
        Returns the path of the project repository

        Uses the global REPOSITORY_NAME constant and searches through parent directories
        """
        p = Path(__file__).absolute().parents
        for parent in p:
            if parent.name == REPOSITORY_NAME:
                return parent

    p = getRepositoryPath()
    checkpoints_folder = p / 'checkpoints'
    generation_folder = checkpoints_folder / 'Generation'

    return generation_folder

def extractAccPpl(read_file_path, write_acc_file_path, write_ppl_file_path):
	with open(write_acc_file_path, "w") as write_acc_file:
		with open(write_ppl_file_path, "w") as write_ppl_file:
			with open(read_file_path, "r") as read_file:
				for line in read_file.readlines():

					results = re.findall(r'.*?Step (\d+?)/.+? acc:  (\d*\.\d*); ppl: (\d*\.\d*);', line)

					if len(results) == 0: continue

					step, acc, ppl = results[0]

					write_acc_file.write(f"{step},{acc}\n")
					write_ppl_file.write(f"{step},{ppl}\n")


generation_folder = setupCheckpoints() / sys.argv[1]
input_file_path = generation_folder / sys.argv[2]
write_acc_file_path = generation_folder / 'accuracy.txt'
write_ppl_file_path = generation_folder / 'perplexity.txt'

extractAccPpl(input_file_path, write_acc_file_path, write_ppl_file_path)
