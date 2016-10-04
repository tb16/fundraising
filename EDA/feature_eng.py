
import pandas as pd
from itertools import izip

def featured(filename):
    '''
    Input: filename
    Output: file with features
    '''

    df = pd.read_csv(filename)
    print df.head()
    df2 = df[df.story.notnull()]
    df2.reset_index(inplace = True)
    df2['title_length'] = df2.title.map(lambda x: len(x))
    df2['story_length'] = df2.story.map(lambda x: len(x))
    df3 = df2.drop(['Unnamed: 0', 'index'], axis = 1)

    df3['percentage'] = [round(x/y, 2) for (x, y) in izip(df3.raised, df3.goal)]
    df3['average_contribution'] = [int(x/y) for (x, y) in izip(df3.raised, df3.people)]
    df3.to_csv('../data/featured_data.csv')

if __name__ == '__main__':
    featured('../data/preprocessed.csv')
