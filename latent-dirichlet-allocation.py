# Thank you to Shashank Kapadia on Medium.com for the guidance

import pandas as pd
import numpy as np
import re
import collections
import gensim
import gensim.corpora as corpora
import nltk
import pyLDAvis
import pyLDAvis.gensim
import pickle

from pprint import pprint
from gensim.utils import simple_preprocess
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud

# SETTINGS -----------------

SUBJECT = 'science'
MIN_WORD_LEN = 3 # minimum number of characters
NUM_TOPICS = 10

# --------------------------

nltk.download('stopwords')
stop_words = stopwords.words('english')

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True, min_len=MIN_WORD_LEN))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc), min_len=MIN_WORD_LEN)
             if word not in stop_words] for doc in texts]

def main():
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
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, width = 1200, height = 800, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud_name = f'wordcloud_{SUBJECT}.png'
    wordcloud.to_file(wordcloud_name)

    data = questions.question_text_processed.values.tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)
    print(data_words[:1][0][:30])

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    print(corpus[:1][0][:30])

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=NUM_TOPICS)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    LDAvis_filename = f'{SUBJECT}_data-{NUM_TOPICS}_topics.html'

    if 1 == 1:
        # mds='mmds' avoids pyLDAvis's default PCoA eigendecomposition, which can
        # produce complex eigenvalues (and a non-JSON-serializable result) with
        # newer numpy/scipy.
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds')
        with open(LDAvis_filename, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk with pickle
    with open(LDAvis_filename, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, LDAvis_filename)


if __name__ == '__main__':
    main()

"""# failed attempt with. old tokenization method
_stemmer = SnowballStemmer('english')
_pattern = r'[a-z]+'
TOKEN_RE = re.compile(_pattern)

STEM_TO_WORD = collections.defaultdict(collections.Counter)

def tokenize(doc):
    tokens = []
    for t in TOKEN_RE.findall(doc):
        if t in ENGLISH_STOP_WORDS:
            continue
        elif len(t) >= MIN_WORD_LEN:
            stem = _stemmer.stem(t)
            STEM_TO_WORD[stem][t] += 1
            tokens.append(stem)
    return tokens"""
