import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from wordcloud import WordCloud

# SETTINGS -----------------

SUBJECT = 'math'
MIN_WORD_LEN = 3         # minimum number of characters per token
MAX_DF = 0.9              # ignore terms that appear in more than this fraction of docs
MIN_DF = 5                 # ignore terms that appear in fewer than this many docs
NMF_COMPONENTS = 10       # intermediate latent components discovered by NMF
NUM_TOPICS = 8             # final number of topics produced by LDA (only used when CLUSTER_SOURCE == 'lda')
NUM_TOP_WORDS = 10         # keywords reported per topic
RANDOM_STATE = 100
CLUSTER_SOURCE = 'nmf_w'    # 'lda'   = LDA is fit on the reconstructed matrix (W @ H), NMF's non-negative
                           #           approximation of the original TF-IDF matrix
                           # 'nmf_w' = LDA is fit on only NMF's W (document-component) matrix

# --------------------------

stop_words = stopwords.words('english') # should we additionally include more stopwords even despite worsened science results?

def clean_text(text):
    text = re.sub('[,.△°∠≠≅¯+=><≤≥|#$%*÷:;_(){}•·!?-]', '', text)
    return text.lower()


def topic_entropy_bits(probs):
    """Shannon entropy (bits) of a document's topic distribution.
    Low = confidently assigned to one topic, high = spread across several topics."""
    probs = np.asarray(probs)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def top_words_for_row(row, feature_names, num_words):
    top_indices = np.argsort(row)[::-1][:num_words]
    return [(feature_names[i], float(row[i])) for i in top_indices]


def main():
    # read CSV
    input_csv = f'{SUBJECT}_questions.csv'
    questions = pd.read_csv(input_csv)
    print(questions)

    # keep the identifying columns around so they can be joined back onto the
    # topic assignments later, since the next step drops them from `questions`
    id_columns = questions[['qid', 'test_id', 'source_file', 'type']].copy()

    questions = questions.drop(columns=['source_file', 'test_id', 'qid', 'type', 'raw_text'])
    questions['question_text_processed'] = questions['clean_text'].map(clean_text)
    print(questions['question_text_processed'].head())

    # Word cloud of the cleaned corpus
    long_string = ' '.join(questions['question_text_processed'].values)
    wordcloud = WordCloud(background_color='white', max_words=5000, contour_width=3,
                           width=1200, height=800, contour_color='steelblue')
    wordcloud.generate(long_string)
    wordcloud_name = f'{SUBJECT}_combined_wordcloud_{CLUSTER_SOURCE}.png'
    wordcloud.to_file(wordcloud_name)
    print(f'Wrote word cloud to {wordcloud_name}')

    documents = questions['question_text_processed'].tolist()

    # TF-IDF
    token_pattern = rf'(?u)\b[a-zA-Z]{{{MIN_WORD_LEN},}}\b'
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=MAX_DF, min_df=MIN_DF,
                                        token_pattern=token_pattern)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print(f'TF-IDF matrix: {tfidf_matrix.shape[0]} documents x {tfidf_matrix.shape[1]} terms')

    # NMF on the TF-IDF matrix
    nmf_model = NMF(n_components=NMF_COMPONENTS, init='nndsvd', random_state=RANDOM_STATE,
                     max_iter=500)
    nmf_doc_component = nmf_model.fit_transform(tfidf_matrix)
    nmf_component_term = nmf_model.components_
    print(f'NMF reconstruction error: {nmf_model.reconstruction_err_:.4f}')

    # LDA on top of either the reconstructed or the W matrix
    if CLUSTER_SOURCE == 'nmf_w':
        lda_input = nmf_doc_component
    elif CLUSTER_SOURCE == 'lda':
        lda_input = nmf_doc_component @ nmf_component_term
    else:
        raise ValueError(f"Unknown CLUSTER_SOURCE: {CLUSTER_SOURCE!r}, expected 'lda' or 'nmf_w'")

    lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, learning_method='batch',
                                           max_iter=100, random_state=RANDOM_STATE)
    final_doc_topic = lda_model.fit_transform(lda_input)
    print(f'LDA perplexity: {lda_model.perplexity(lda_input):.4f}')

    if CLUSTER_SOURCE == 'nmf_w':
        # LDA was fit on the component space, so project its topics back onto
        # vocabulary terms through the NMF term basis to describe them with words.
        final_topic_term = lda_model.components_ @ nmf_component_term  # (NUM_TOPICS, n_terms)
    else:
        # LDA was fit directly on the reconstructed term-space matrix, so its
        # topics are already expressed over vocabulary terms.
        final_topic_term = lda_model.components_  # (NUM_TOPICS, n_terms)
    final_num_topics = NUM_TOPICS

    # Per-question topic assignments
    dominant_topics = final_doc_topic.argmax(axis=1)
    dominant_probs = final_doc_topic.max(axis=1)
    entropies = [topic_entropy_bits(row) for row in final_doc_topic]
    topic_distributions = [
        '|'.join(f'{tid}:{prob:.4f}' for tid, prob in enumerate(row))
        for row in final_doc_topic
    ]

    questions_topics_df = pd.DataFrame({
        'qid': id_columns['qid'].values,
        'test_id': id_columns['test_id'].values,
        'source_file': id_columns['source_file'].values,
        'type': id_columns['type'].values,
        'question_text': questions['clean_text'].values,
        'dominant_topic': dominant_topics,
        'dominant_topic_probability': dominant_probs,
        'topic_entropy_bits': entropies,
        'topic_distribution': topic_distributions,
    })
    questions_topics_csv = f'{SUBJECT}_combined_questions_topics_{CLUSTER_SOURCE}.csv'
    questions_topics_df.to_csv(questions_topics_csv, index=False)
    print(f'Wrote {len(questions_topics_df)} question-topic assignments to {questions_topics_csv}')

    # Final topic summaries
    num_docs = len(questions_topics_df)
    topics_rows = []
    for topic_id, row in enumerate(final_topic_term):
        top = top_words_for_row(row, feature_names, NUM_TOP_WORDS)
        words = [w for w, _ in top]
        weights = [w for _, w in top]
        docs_dominant = int((questions_topics_df['dominant_topic'] == topic_id).sum())
        topics_rows.append({
            'topic_id': topic_id,
            'top_keywords': ', '.join(words),
            'keyword_weights': ', '.join(f'{w:.4f}' for w in weights),
            'num_documents_dominant': docs_dominant,
            'proportion_documents_dominant': docs_dominant / num_docs if num_docs else 0.0,
            'avg_probability_across_corpus': float(final_doc_topic[:, topic_id].mean()),
            'num_topics': final_num_topics,
            'nmf_components': NMF_COMPONENTS,
        })
    topics_df = pd.DataFrame(topics_rows)
    topics_csv = f'{SUBJECT}_combined_topics_{CLUSTER_SOURCE}.csv'
    topics_df.to_csv(topics_csv, index=False)
    print(f'Wrote {len(topics_df)} topic summaries to {topics_csv}')

    # Intermediate NMF component summaries (diagnostic)
    nmf_rows = []
    for component_id, row in enumerate(nmf_component_term):
        top = top_words_for_row(row, feature_names, NUM_TOP_WORDS)
        nmf_rows.append({
            'nmf_component_id': component_id,
            'top_keywords': ', '.join(w for w, _ in top),
            'keyword_weights': ', '.join(f'{w:.4f}' for _, w in top),
        })
    nmf_components_df = pd.DataFrame(nmf_rows)
    nmf_components_csv = f'{SUBJECT}_combined_nmf_components_{CLUSTER_SOURCE}.csv'
    nmf_components_df.to_csv(nmf_components_csv, index=False)
    print(f'Wrote {len(nmf_components_df)} NMF component summaries to {nmf_components_csv}')

    doc_topic_dist = final_doc_topic
    perplexity = min(30, len(doc_topic_dist) - 1)
    tsne = TSNE(n_components=2, random_state=100, perplexity=perplexity, init='pca')
    tsne_embeddings = tsne.fit_transform(doc_topic_dist)
    # Organize data for plotting
    plot_df = pd.DataFrame(tsne_embeddings, columns=['x', 'y'])
    plot_df['dominant_topic'] = np.argmax(doc_topic_dist, axis=1)

    # Scatter plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        plot_df['x'],
        plot_df['y'],
        c=plot_df['dominant_topic'],
        cmap='tab10',
        alpha=0.6,
        edgecolors='none'
    )
    plt.colorbar(scatter, label='Dominant Topic ID')
    title_source = 'W Matrix' if CLUSTER_SOURCE == 'nmf_w' else 'Reconstructed Matrix'
    plt.title(f'LDA on {title_source} Topics Visualized via t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(f'LDA_visualization_{SUBJECT}_{CLUSTER_SOURCE}.png')


if __name__ == '__main__':
    main()
