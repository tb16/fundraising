
import numpy as np
import pandas as pd
from itertools import izip
import count_length

def featured(filename):
    '''
    Input: filename
    Output: file with features
    '''

    df = pd.read_csv(filename)
    # remove row which doesn't have story. and set the new index
    df2 = df[df.story.notnull()]
    df2.reset_index(inplace = True)
    # calculate characters and words length in title and story
    df2['title_length']  = df2.title.map(lambda x: len(x))
    df2['story_length'] = df2.story.map(lambda x: len(x))
    df2['word_count_title'] = df2.title.map(lambda x: count_length.count_words)
    df2['word_count_story'] = df2.story.map(lambda x: count_length.count_words)
    df2['sentence_count_story'] = df2.story.map(lambda x: count_length.count_sentences)
    df3 = df2.drop(['Unnamed: 0', 'index'], axis = 1)

    # Calculate percentage, average_contribution
    df3['percentage'] = [round(x/y, 2) for (x, y) in izip(df3.raised, df3.goal)]
    df3['average_contribution'] = [int(x/y) for (x, y) in izip(df3.raised, df3.people)]

    # remove the outliers
    df4 = df3[(df3.average_contribution > 0) & (df3.goal>10) & (df3.percentage < 100) \
    & (df3.average_contribution < 5000)]
    df4.to_csv('../data/featured_data2.csv')
    return None


def over_or_underfunded(df):
    '''
    counts the number of overfunded and underfunded case at different thresholds
    Input: dataframe
    Output: overfunded or underfunded at different threshold percentages
    '''

    for i in np.linspace(0.2, 2, 10):
        success = (df.percentage > i).sum()

        print i
        print 'success case: ', success
        print 'underfunded: ',len(df)- success

if __name__ == '__main__':
    df = featured('../data/preprocessed.csv')
    # over_or_underfunded(df)
