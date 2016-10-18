import requests
import re
import time
import bs4
import pandas as pd
import cPickle as pickle
from datetime import datetime

category_list = [\
'Medical','Medical-Illness-Healing',
'Memorials',
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




url = 'https://www.gofundme.com/2t9bhces'
_user_agent = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36')


df = pd.DataFrame(index=range(0,1),columns=['category'], dtype='string')
with open('../data/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)
with open('../data/nmf_model.pkl') as f:
    nmf_model = pickle.load(f)
with open('../data/mnb_model.pkl') as f:
    mnb_model = pickle.load(f)


def download(url, *a, **kw):
    '''
    download and returns the html parsed beautifulsoup
    '''
    _user_agent = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36')
    kw.setdefault('headers', {})['User-Agent'] = _user_agent
    return bs4.BeautifulSoup(requests.get(url, *a, **kw).text, 'html.parser')


def search_facebook(title):
    '''
    url search for fb friends gofund me website given a title
    '''
    search_url = 'https://www.gofundme.com/mvc.php?'
    soup = download(search_url, params={'term' : title, 'route': 'search'})
    for tile in soup.select('.search_tile'):
        try:
            link = 'https:'+tile.select('.name')[0]['href']
            if tile.select('.title')[0].text.strip() == title:
                friends = tile.select('.fb_count')[0].text
                num_friend = get_number(friends.replace('Facebook Friends', '').strip())
                return num_friend

        except:
            continue
    return 700.



def get_number(s):
    factor = 1
    if 'k' in s or 'K' in s:
        factor = 1000
    s = float(''.join(c for c in s if c.isdigit() or c == '.'))
    return round(s*factor, 2)


def get_goal(s):
    factor = 1
    if 'k' in s or 'K' in s:
        factor = 1000
    if 'm' in s or 'M' in s:
        factor = 1000000
    s = float(''.join(c for c in s if c.isdigit() or c == '.'))
    return round(s*factor, 2)

def get_data(url):

    soup = bs4.BeautifulSoup(requests.get(url, _user_agent).text, 'html.parser')
    # raised amount and goal
    raised = soup.select('.goal strong')
    if raised:
        raised = get_number(raised[0].text)
    goal = soup.select('.goal .smaller')[0].text
    goal = re.findall(r'of (.*?) goal', goal)
    goal = get_goal(goal[0])
    # title and story
    title = soup.find(class_='campaign-title').text.strip()
    story =  soup.find(class_='co-story').text.strip()
    # people
    status = soup.select('.campaign-status.text-small')[0]
    people = status.find('span').text
    people = get_number(people)
    # name_place
    details = soup.select('.co-details')[0]
    if details:
        place = details.find('a').text.strip()
        details.find('a').extract()
        name = details.text.strip()

    else:
        name = ''
        place = ''
    # category
    category = soup.select('.pills-contain a')
    if category:
        category = category[0].text.strip()
    else:
        category = 'Other'

    # shares
    share = soup.select('.js-share-count-text')
    if share:
        shares = get_number(share[0].text.strip())
    else:
        shares = 0

    # date_created/recorded/days
    created = soup.select('.created-date')[0].text.strip()
    date = re.findall('Created (.*)', created)[0]
    time_now = time.time()
    date_struct_created = datetime.strptime(date, "%B %d, %Y")
    date_struct_recorded = datetime.fromtimestamp(time_now)
    date_created = date_struct_created.strftime('%Y-%m-%d')
    date_recorded = date_struct_recorded.strftime('%Y-%m-%d')
    days = (date_struct_recorded - date_struct_created).days
    # friends
    friends = search_facebook(title)
    # data = [category, friends, name, place, story, title, date_created, \
    # date_recorded, days, raised, goal, shares, people]
    df['category'] = category
    df['friends'] = int(friends)
    df['name'] = name
    df['place'] = place
    # unicode 'story'
    x = story.encode("utf-8")
    y = unicode(x, errors = 'ignore')
    df['story'] =y
    # df['story'] = unicode(story, errors = 'ignore')
    # unicode titlex = story.encode("utf-8")
    x1 = title.encode("utf-8")
    y1 = unicode(x1, errors = 'ignore')
    df['title'] = y1
    df['date_created'] = date_created
    df['date_recorded'] = date_recorded
    df['days'] = days
    df['raised'] = raised
    df['goal'] = goal
    df['shares'] = int(shares)
    df['people'] = people



    #cleaning
    df['friends_share'] = friends/shares
    df['month_created'] = datetime.strptime(date_created, '%Y-%m-%d').month
    df['title_length'] = len(title)
    df['story_length'] = len(story)
    df['word_count_title'] = len(title.split())
    df['word_count_story'] = len(story.split())
    df['sentence_count_story'] = len(story.split('.'))
    df['percentage'] = round(raised/goal, 2)
    df['average_contribution'] = raised/people

    return df


def featurizing(df):
    # category split
    for cat in sorted(category_list):
        if cat == df.category[0]:
            df[cat] = 1
        else:
            df[cat] = 0


    vector = vectorizer.transform(df.story)
    nmf_topics = nmf_model.transform(vector)
    col_name = ['nmf_topic_' + str(i) for i in xrange(0,10)]
    df2 = pd.DataFrame(nmf_topics, columns = col_name)
    df = pd.concat((df, df2), axis = 1)
    df['mnb_probs'] = mnb_model.predict_proba(vector)[:,1]
    y = int(df.percentage[0] > 0.5)
    drop_list = ['category', 'name','place','story','title','date_created','date_recorded','raised','people','percentage',\
                'average_contribution']

    df.drop(drop_list, axis =1, inplace = True)
    X = df.values
    return X, y, df

if __name__ == '__main__':
    df = get_data(url)
    X, y, df = featurizing(df)
