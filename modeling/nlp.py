from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB as mnb
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import cPickle as pickle
from tokenizer import tokenize
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import (cluster, decomposition, ensemble, preprocessing)

with open('../data/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)
with open('../data/sparse_mat.pkl') as f:
    sparse_mat = pickle.load(f)


df = pd.read_csv('../data/featured_data.csv')

docs = df.story
y = df.percentage > 0.5
X_train, X_test, y_train, y_test = train_test_split(docs, \
            y, test_size = 0.3, random_state = 1234)
X_train1, X_train2,y_train1, y_train2 = train_test_split(X_train, y_train, \
                                                         test_size = 0.8, random_state = 1234)


def mnb_model():
    '''

    '''

    X_vect = vectorizer.transform(X_train1)
    clf = mnb()
    clf.fit(X_vect, y_train1)
    probs = clf.predict_proba(doc_vect)[:, 1]
    score = cross_val_score(clf, X_vect, y_train1,cv=5)
    return clf, probs, score.mean()


def kmeans():
    '''
    returns the kmeans clusters model and cluster labels for all documents
    '''
    kmeans = KMeans(n_clusters = 10,n_jobs = -1 )
    X_vect = vectorizer.transform(docs)
    clusters = kmeans.fit(X_vect)
    labels = clusters.labels_
    return clusters, labels


def scree_plot(num_components, pca):
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35,
           color=[(0.949, 0.718, 0.004),
                  (0.898, 0.49, 0.016),
                  (0.863, 0, 0.188),
                  (0.694, 0, 0.345),
                  (0.486, 0.216, 0.541),
                  (0.204, 0.396, 0.667),
                  (0.035, 0.635, 0.459),
                  (0.486, 0.722, 0.329),
                 ])
    for i in xrange(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

    ax.set_xticklabels(ind,
                       fontsize=12)

    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, 8+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    plt.title("Scree Plot", fontsize=16)


def pca_model(vector):
    '''
    Input : vector eg vectorized docs
    Output : pca model and standard scalar
    '''
    ss = preprocessing.StandardScaler()
    X_centered = ss.fit_transform(vector)
    pca = decomposition.PCA(n_components=1000)
    pca.fit_transform(X_centered)
    return pca, ss


def nmf_model(n_topics = 10):
    '''
    Takes the vectorized document and returns the topic label.
    '''
    nmf = decomposition.NMF(n_components=n_topics, random_state=1,
              alpha=.1, l1_ratio=.5).fit(doc_vect)

    topics = nmf.transform(doc_vect).argsort()
    topic_labels = [topic[-1] for topic in topics]
    return nmf, topic_labels


def print_top_words(model, feature_names, n_top_words = 20):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[-1:-n_top_words - 1:-1]]))



if __name__ == '__main__':
    doc_vect = sparse_mat
    nmf_model, topic_labels = nmf_model(doc_vect)
    print  'nmf done'
    mnb_model, probs, score = mnb_model()
    with open('../data/nmf_model.pkl', 'w') as f:
        pickle.dump(nmf_model, f)
    with open('../data/mnb_model.pkl', 'w') as f:
        pickle.dump(mnb_model, f)

# To print out the topic words in nmf
# feature_names = vectorizer.get_feature_names()
# print_top_words(nmf, feature_names)
