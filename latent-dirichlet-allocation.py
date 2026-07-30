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
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from wordcloud import WordCloud

# SETTINGS -----------------

SUBJECT = 'reading'
MIN_WORD_LEN = 3 # minimum number of characters
NUM_TOPICS = 8

# --------------------------

stop_words = stopwords.words('english')

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True, min_len=MIN_WORD_LEN))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc), min_len=MIN_WORD_LEN)
             if word not in stop_words] for doc in texts]

def get_document_topics_full(lda_model, corpus):
    """Full topic-probability distribution for every document (no thresholding)."""
    return [lda_model.get_document_topics(bow, minimum_probability=0) for bow in corpus]

def topic_entropy_bits(topic_probs):
    """Shannon entropy (bits) of a document's topic distribution.
    Low = confidently assigned to one topic, high = spread across several topics."""
    probs = np.array([p for _, p in topic_probs])
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def main():
    # read CSVs
    input_csv = f'{SUBJECT}_questions.csv'
    questions = pd.read_csv(input_csv)
    print(questions)

    # keep the identifying columns around so they can be joined back onto the
    # topic assignments later, since the next step drops them from `questions`
    id_columns = questions[['qid', 'test_id', 'source_file', 'type']].copy()

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
                                           num_topics=NUM_TOPICS,random_state=100,
                                           chunksize=100,
                                           passes=50)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())

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

    # --- Additional outputs: per-question topic assignments & per-topic summaries ---
    doc_topics_full = get_document_topics_full(lda_model, corpus)

    dominant_topics = []
    dominant_probs = []
    entropies = []
    topic_distributions = []
    for topic_probs in doc_topics_full:
        dominant_topic_id, dominant_prob = max(topic_probs, key=lambda x: x[1])
        dominant_topics.append(dominant_topic_id)
        dominant_probs.append(dominant_prob)
        entropies.append(topic_entropy_bits(topic_probs))
        topic_distributions.append(
            '|'.join(f'{tid}:{prob:.4f}' for tid, prob in topic_probs))

    questions_topics_df = pd.DataFrame({
        'qid': id_columns['qid'].values,
        'test_id': id_columns['test_id'].values,
        'source_file': id_columns['source_file'].values,
        'type': id_columns['type'].values,
        'question_text': questions['clean_text'].values,
        'word_count': [len(doc) for doc in data_words],
        'dominant_topic': dominant_topics,
        'dominant_topic_probability': dominant_probs,
        'topic_entropy_bits': entropies,
        'topic_distribution': topic_distributions,
    })
    questions_topics_csv = f'{SUBJECT}_questions_topics.csv'
    questions_topics_df.to_csv(questions_topics_csv, index=False)
    print(f'Wrote {len(questions_topics_df)} question-topic assignments to {questions_topics_csv}')

    # Per-topic summary, including coherence/perplexity model metrics
    coherence_model = CoherenceModel(model=lda_model, texts=data_words,
                                      dictionary=id2word, coherence='c_v')
    per_topic_coherence = coherence_model.get_coherence_per_topic()
    overall_coherence = coherence_model.get_coherence()
    model_log_perplexity = lda_model.log_perplexity(corpus)

    num_docs = len(questions_topics_df)
    topics_rows = []
    for topic_id, word_weight_pairs in lda_model.show_topics(num_topics=NUM_TOPICS, num_words=10, formatted=False):
        words = [w for w, _ in word_weight_pairs]
        weights = [wt for _, wt in word_weight_pairs]
        docs_dominant = int((questions_topics_df['dominant_topic'] == topic_id).sum())
        avg_prob_corpus = float(np.mean(
            [dict(doc).get(topic_id, 0.0) for doc in doc_topics_full]))
        topics_rows.append({
            'topic_id': topic_id,
            'top_keywords': ', '.join(words),
            'keyword_weights': ', '.join(f'{w:.4f}' for w in weights),
            'num_documents_dominant': docs_dominant,
            'proportion_documents_dominant': docs_dominant / num_docs if num_docs else 0.0,
            'avg_probability_across_corpus': avg_prob_corpus,
            'coherence_c_v': per_topic_coherence[topic_id],
            'model_overall_coherence_c_v': overall_coherence,
            'model_log_perplexity': model_log_perplexity,
            'num_topics': NUM_TOPICS,
        })

    topics_df = pd.DataFrame(topics_rows).sort_values('topic_id').reset_index(drop=True)
    topics_csv = f'{SUBJECT}_topics.csv'
    topics_df.to_csv(topics_csv, index=False)
    print(f'Wrote {len(topics_df)} topic summaries to {topics_csv}')


if __name__ == '__main__':
    main()
