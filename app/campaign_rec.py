import pandas as pd
import numpy as np
from string import punctuation
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tokenizer import tokenize


df = pd.read_csv('../data/featured_data.csv')

def bag_of_words(df):
    '''
    Applies Tfidf vectorizer to descriptions in the
    dataframe.
    Returns the vectorizer instance and sparse matrix.
    '''
    vectorizer = TfidfVectorizer(max_features = 4000, decode_error='ignore', max_df = 0.90, min_df= 2, stop_words = 'english', tokenizer = tokenize)
    vectorizer.fit(df.story)
    sparse = vectorizer.fit_transform(df.story)
    return vectorizer, sparse



def pickle_vec(vectorizer, sparse):
    '''
    Pickle the vectorizer instance and sparse matrix
    '''
    v = open('../data/vectorizer.pkl', 'w')
    pickle.dump(vectorizer, v)
    v.close()

    s = open('../data/sparse_mat.pkl', 'w')
    pickle.dump(sparse, s)
    s.close()


def get_success_index(df):
    '''
    returns the indices of successsful campaigns from the dataframe
    didn't use it
    '''
    indices = df[df.percentage>=0.5].index.tolist()
    return np.array(indices)


def similar_campaign(vector, vectorizer, sparse_mat):
    '''
    Finds the similar success story to the given campaign. returns top 3 campaigns
    and keywords. similarity from cosine similarity with tfidf vectors. top words
    from tfidf values of a story
    '''


    feature_names = np.array(vectorizer.get_feature_names())

    similarity = linear_kernel(vector, sparse_mat)
    top_indices_story = np.argsort(similarity.flatten())[-1::-1]
    success_indices = []
    for top_index in top_indices_story:
        if df.percentage[top_index] >= 0.5:
            success_indices.append(top_index)
    keywords = []
    for idx in success_indices[:3]:
        keywords_indices = np.argsort(sparse_mat[idx].toarray()).flatten()[-1:-11:-1]
        keywords.append(' '.join(feature_names[keywords_indices]))
    output_df = df.iloc[success_indices[:3]]
    output_df['keywords'] = keywords
    return output_df[['category', 'days','title', 'story', 'friends','shares', 'goal', 'percentage', 'keywords']]


if __name__ == '__main__':

    # df = df[df['percentage'] >= 0.50]
    # df.to_csv('../data/featured_data_success.csv', index = False)
    vectorizer, sparse = bag_of_words(df)
    pickle_vec(vectorizer, sparse)
