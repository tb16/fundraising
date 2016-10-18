
import pymongo
from collections import defaultdict
import pandas as pd
import numpy as np
import re
from datetime import datetime
from itertools import izip


def unique_data(database_name, table_name):
    '''
    Input: MongoDB database, collections
    Output: Unique documents and save into csv file in the data folder
    '''

    db = pymongo.MongoClient()[database_name]
    cursor = db[table_name]
    df = pd.DataFrame(list(cursor.find()))
    items = list(cursor.find())
    groups = defaultdict(list)
    keep = []

    for item in items:
        groups[item['story']].append(item)


    for title, items in groups.iteritems():
        good = sorted(items, key=lambda x: x['download_time'])[-1]
        keep.append(good)
    df = pd.DataFrame(keep)
    df2 = df[df.story.notnull()]
    return df


def set_category(url):
    '''
    input: string: category_url, eg: 'https://www.gofundme.com/Competitions-Pageants/'
    output: string: category, eg : 'Competitions-Pageants'
    '''
    return url.split('/')[-2]


def date_conversion(created = None, recorded = None):
    '''
    Input : string: created is the date_string in the format: 'Created September 16, 2016'
            float: recorded is the timestamp of downloaded time
    Output: tuple of datetime struct, and string in the format:'2016-09-16'
    '''

    if created:
        date_created_extract = re.findall('Created (.*)',  created.strip())[0]
        date_struct = datetime.strptime(date_created_extract, "%B %d, %Y")
    elif recorded:
        date_struct = datetime.fromtimestamp(recorded)
    return date_struct, date_struct.strftime('%Y-%m-%d')


def num_days(created_date, record_date):
    '''
    Input : Date, Date
    Output : difference in days
    '''
    delta = date_conversion(recorded = record_date)[0]- date_conversion(created = created_date)[0]
    return delta.days


def money_raised_and_goal(df):
    '''
    Input : DataFrame
    Output : None
    Separates the dataframes money column, from dictionary into separate columns \
    of raised amount and goal amount.
    '''

    df['raised'] = df.money.map(lambda x: x['raised'])
    df['goal'] = df.money.map(lambda x: x['goal'])


def num_contributor(df):
    '''
    Input: DataFrame
    Output : None
    '''

    df['num_contributor'] = df.status.map(lambda x: x['people'])


def num_people(status):
    '''
    Input : string , eg. '1,234'
    Output : returns the integer values : eg 1234
    '''
    number = status['people']
    return int(number.replace(',',''))


if __name__ == '__main__':
    df = unique_data('gfm_database', 'campaigns')
    # set the category from category url
    cond = df.category.isnull() & df.category_url
    df.category[cond] = df.category_url[cond].map(set_category)
    # set the date created
    df['date_created'] = df.date.map(lambda x: date_conversion(created = x)[1])
    # set the date_recorded
    df['date_recorded'] = df.download_time.map(lambda x: date_conversion(recorded = x)[1])
    # calculate_days
    df['days'] = [num_days(x, y) for (x, y) in izip(df.date, df.download_time)]
    # money raised and goal
    money_raised_and_goal(df)
    # Number of shares
    df['shares'] = df.share.map(lambda x: float(str(x)[:-1])*1000 if str(x)[-1] == 'K' else float(x))
    df['people'] = df.status.map(num_people)
    # remove first row which doesn't have story. and set the new index
    df2 = df.iloc[1:, :]
    # changing to unicode:
    df2.story = map(lambda x: unicode(x, errors = 'ignore'), df2.story)
    # df2.reset_index(inplace = True)
    drop_list = ['_id','category_url','date','download_time', 'money', 'share', 'status']
    df2.drop(drop_list, axis = 1, inplace = True)
    # Calculate percentage, average_contribution
    df2['percentage'] = [round(x/y, 2) for (x, y) in izip(df2.raised, df2.goal)]
    df2['average_contribution'] = [int(x/y) for (x, y) in izip(df2.raised, df2.people)]
    # remove the outliers
    df3 = df2[(df2.average_contribution > 0) & (df2.goal>500) & (df2.percentage < 3) \
    & (df2.average_contribution < 5000)]
    df3.to_csv('../data/preprocessed.csv', encoding='utf-8', index = False)
