
import numpy as np
import pandas as pd
from itertools import izip
import count_length
from datetime import datetime


category_list = [\
'Medical','Medical-Illness-Healing',
'Memorials','Funerals-Memorials_Tributes',
'Education','Education-Schools-Learning',
'Emergencies',
'Charity','Non-Profits-Charities',
'Sports','Sports-Teams-Clubs',
'Competitions','Competitions-Pageants',
'Animals','Animals-Pets',
'Volunteer','Volunteer-Service',
'Creative','Creative-Arts-Music-Film',
'Community','Community-Neighbors',
'Business','Business-Entrepreneurs',
'Events','Celebrations-Special-Events',
'Wishes','Dreams-Hopes-Wishes',
'Faith','Missions-Faith-Church',
'Travel','Travel-Adventure',
'Family','Babies-Kids-Family',
'Accidents-Personal-Crisis',
'Weddings-Honeymoons',
'Other',
]

def featured(filename):
    '''
    save and return the dataframe after feature engineering.
    Input: filename
    Output: dataframe with feature
    '''

    df = pd.read_csv(filename)

    df['category']=df.category.map(lambda x: x if x in category_list else 'Other')
    # Extracting the month of funding camaign
    df['month_created'] = df.date_created.map(lambda x: datetime.strptime(x, '%Y-%m-%d').month)
    # calculate characters and words length in title and story
    df['title_length']  = df.title.map(lambda x: len(x))
    df['story_length'] = df.story.map(lambda x: len(x))
    df['word_count_title'] = df.title.map(lambda x: count_length.count_words(x))
    df['word_count_story'] = df.story.map(lambda x: count_length.count_words(x))
    df['sentence_count_story'] = df.story.map(lambda x: count_length.count_sentences(x))
    # Replace '0' friend counts with mean:
    friends_mean = int((df.loc[df.friends > 0,'friends']).mean())
    df.loc[df.friends == 0,'friends'] = friends_mean
    # Replace '0' share counts with mean:
    shares_mean = int((df.loc[df.shares > 0,'shares']).mean())
    df.loc[df.shares == 0,'shares'] = friends_mean
    # Calculate ratio of friends to share
    df['friends_share']= df.friends/df.shares



    df.to_csv('../data/featured_data.csv', index = False)
    return df


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
