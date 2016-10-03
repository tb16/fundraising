
import pymongo
from collections import defaultdict


def unique_data():
    '''
    Input: MongoDB collections
    Output: Unique documents 

    db = pymongo.MongoClient().gfm_database
    cursor = db.campaigns
    df = pd.DataFrame(list(cursor.find()))



    items = list(cursor.find())
    groups = defaultdict(list)
    keep = []

    for item in items:
        groups[item['story']].append(item)

    for title, items in groups.iteritems():
        good = sorted(items, key=lambda x: x['download_time'])[-1]
        keep.append(good)

    return keep
