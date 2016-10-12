
import numpy as np
import pandas as pd
from itertools import izip
import count_length
from category import category_list as cls
from datetime import datetime


def featured(filename):
    '''
    Input: filename
    Output: file with features
    '''

    df = pd.read_csv(filename)
    # remove row which doesn't have story. and set the new index
    df2 = df[df.story.notnull()]
    # df2.reset_index(inplace = True)
    # Replace '0' friend counts with mean:
    friends_mean = int((df.loc[df.friends > 0,'friends']).mean())
    df2.loc[df2.friends == 0,'friends'] = friends_mean
    # Replace '0' share counts with mean:
    shares_mean = int((df.loc[df.shares > 0,'shares']).mean())
    df2.loc[df2.shares == 0,'shares'] = friends_mean
    # Calculate ratio of friends to share
    df2['friends_share']= df2.friends/df2.shares
    # Extracting the month of funding camaign
    df2['month_created'] = df.date_created.map(lambda x: datetime.strptime(x, '%Y-%m-%d').month)
    # calculate characters and words length in title and story
    df2['title_length']  = df2.title.map(lambda x: len(x))
    df2['story_length'] = df2.story.map(lambda x: len(x))
    df2['word_count_title'] = df2.title.map(lambda x: count_length.count_words(x))
    df2['word_count_story'] = df2.story.map(lambda x: count_length.count_words(x))
    df2['sentence_count_story'] = df2.story.map(lambda x: count_length.count_sentences(x))
    # Calculate percentage, average_contribution
    df2['percentage'] = [round(x/y, 2) for (x, y) in izip(df2.raised, df2.goal)]
    df2['average_contribution'] = [int(x/y) for (x, y) in izip(df2.raised, df2.people)]
    df2['category']=df2.category.map(lambda x: x if x in cls() else 'Other')
    # remove the outliers
    df3 = df2[(df2.average_contribution > 0) & (df2.goal>500) & (df2.percentage < 3) \
    & (df2.average_contribution < 5000)]
    df3.to_csv('../data/featured_data.csv', index = False)
    return df3


def over_or_underfunded(df):
    '''
    counts the number of overfunded and underfunded case at different thresholds
    Input: dataframe
    Output: overfunded or underfunded at different threshold percentages
    '''

    for i in np.linspace(0, 2, 11):
        success = (df.percentage > i).sum()
        print i
        print 'success case: ', success
        print 'underfunded: ',len(df)- success

if __name__ == '__main__':
    df = featured('../data/preprocessed.csv')
    # over_or_underfunded(df)
