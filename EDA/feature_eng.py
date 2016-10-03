import pandas as pd
import numpy as np
import pymongo
import random
from datetime import datetime
import time
import re
from collections import defaultdict


def mongo_to_df_unique():
    db = pymongo.MongoClient().gfm_database
    cursor = db.campaigns
    items = list(cursor.find())
    groups = defaultdict(list)
    keep = []
    for item in items:
        groups[item['title']].append(item)
    for title, items in groups.iteritems():
        good = sorted(items, key=lambda x: x['download_time'])[-1]
        keep.append(good)
    return pd.DataFrame(keep)

def time_convert(time_string, download_time):
    record_date = datetime.fromtimestamp(download_time)
    created_date = datetime.strptime(time_string, "%B %d, %Y")
    delta = (record_date-created_date)
    return record_date, created_date, delta.days
def date_format(date):
    return date.strftime('%Y-%m-%d')


def set_category():
    cond = df2.category.isnull()
    df2.category[cond] = df2.category_url[cond].map(lambda x: x.split('/')[-2] if x else None)
def date_convert(df):
    df['date_created'] = df.date.map(lambda x: re.findall('Created (.*)',  x.strip())[0])
def money_raised_and_goal(df):
    df['raised'] = df.money.map(lambda x: x['raised'])
    df['goal'] = df.money.map(lambda x: x['goal'])
def num_contributor(df):
    df['num_contributor'] = df.status.map(lambda x: x['people'])
