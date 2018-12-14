# Python program to generate WordCloud
# Inspired from https://www.geeksforgeeks.org/generating-word-cloud-python/
  
# importing all necessery modules 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
import util
  
reviews, labels, _, _, _, _ = util.load_behavioral_tsv_dataset('../data/large_behavior_balanced.tsv')

real_review_words = ' '
fake_review_words = ' '
stopwords = set(STOPWORDS)

for idx_review, review in enumerate(reviews):
    tokens = review.split()

    for idx_word, word in enumerate(review.split()):
        if labels[idx_review] == 0:
            real_review_words += word.lower() + ' '
        else:
            fake_review_words += word.lower() + ' '
  
real_wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(real_review_words)
fake_wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(fake_review_words)
  

plt.figure(1)                     
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(real_wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.savefig('../checkpoints/NaiveBayes/large_real_wordcloud.png')

plt.figure(2)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(fake_wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.savefig('../checkpoints/NaiveBayes/large_fake_wordcloud.png')