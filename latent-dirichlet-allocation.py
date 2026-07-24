# Thank you to Shashank Kapadia on Medium for the guidance

import pandas as pd
import numpy as np
import matplotlib as plt
import re
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud

# SETTINGS -----------------

SUBJECT = 'reading'

# --------------------------

# read CSVs
input_csv = f'{SUBJECT}_questions.csv'
questions = pd.read_csv(input_csv)
print(questions)

# data cleaning
questions = questions.drop(columns=['source_file', 'test_id', 'qid', 'type', 'raw_text'])
print(questions)

# Remove punctuation
questions['question_text_processed'] = \
questions['clean_text'].map(lambda x: re.sub('[,.△°∠≠≅¯+=><≤≥|#$%*÷:;_(){}•·!?-]', '', x))
# Convert the titles to lowercase
questions['question_text_processed'] = \
questions['question_text_processed'].map(lambda x: x.lower())
# Print out the first rows of questions
print(questions['question_text_processed'].head())

# Join the different processed titles together.
long_string = ','.join(list(questions['question_text_processed'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud_name = f'wordcloud_{SUBJECT}.pn'
wordcloud.to_file(wordcloud_name)