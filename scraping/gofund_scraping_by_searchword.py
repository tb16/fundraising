import requests
import sys
import json
import re
import time
import multiprocessing
import multiprocessing.pool
import bs4
import pymongo

db = pymongo.MongoClient().test_database
# tab = db['campaigns']
_user_agent = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36')

interval = 2.5
lock = multiprocessing.Lock()
last_time = 0


states = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",\
"Delaware","District of Columbia","Florida","Georgia","Hawaii","Idaho",\
"Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Montana",\
"Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",\
"North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Maryland",\
"Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Pennsylvania",\
"Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",\
"Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]




def download(url, *a, **kw):
    kw.setdefault('headers', {})['User-Agent'] = _user_agent
    return bs4.BeautifulSoup(requests.get(url, *a, **kw).text, 'html.parser')


def download_campaign(url):
    global last_time

    with lock:
        now = time.time()
        if now - last_time < interval:
            time.sleep(interval - (now - last_time))
        last_time = time.time()

        return download(url)


def get_money_amount(s):
    return round(float(''.join(c for c in s if c.isdigit() or c == '.')), 2)

def get_facebook_friend(s):
    return int(float(''.join(c for c in s if c.isdigit() or c == '.')))

def get_campaign(url):
    soup = download_campaign(url)
    raised = soup.select('.goal strong')
    if raised:
        raised = get_money_amount(raised[0].text)
    else:
        raised = 0.0
    goal = soup.select('.goal .smaller')[0].text
    goal = re.findall(r'of (.*?) goal', goal)
#     print goal
    goal = get_money_amount(goal[0].replace('k', '000'))
    title = soup.find(class_='campaign-title').text.strip()
    story =  soup.find(class_='co-story').text.strip()

    status = soup.select('.campaign-status.text-small')[0]
    people = status.find('span').text
    days = re.findall('(\d*) day', status.text)
    if len(days)>0:
        days = int(days[0])
    else:
        days = 0

    details = soup.select('.co-details')[0]
    if details:
        location = details.find('a').text.strip()
        details.find('a').extract()
        name = details.text.strip()
        # name = name[0].text.strip()
    else:
        name = ''
        location = ''

# Category check:
    category = soup.select('.pills-contain a')
    if category:
        category = category[0].text.strip()
    else:
        category = ''


    share = soup.select('.js-share-count-text')
    if share:
        share = share[0].text.strip()
    else:
        share = 0
    date = soup.select('.created-date')
    if date:
        date = date[0].text.strip()
    else:
        date = ''

    return {
        'money': {
            'raised': raised,
            'goal': goal
        },
        'title': title,
        'category': category,
        'story': story,
        'name': name,
        'place': location,
        'share': share,
        'date': date,
        'status': {
            'people': people,
            'days': days,
        }
    }


def search_list():
    return states

if __name__ == '__main__':
    search_url = 'https://www.gofundme.com/mvc.php?'
    search_terms = search_list()
    for j, search_term in enumerate(search_terms):
        for i in xrange(1, 51):
            print {'searching': search_term, 'loop': i }
            soup = download(search_url, params={'page': i, 'term' : search_term, 'route': 'search'})
            # print soup

            for tile in soup.select('.search_tile'):

                # print count
                try:
                    # link = tile.select('a.name')[0]
                    # fbcount = tile.select('a.fb_count')
                    link = tile.select('.name')[0]
                    fbcount = tile.select('.fb_count')
                    if fbcount:
                        friends = get_facebook_friend(fbcount[0].text)
                        # friends = int(friends.group())
                    else:
                        friends = 0
                    # print link['href']
                    campaign = get_campaign('https:'+link['href'])
                    campaign['friends'] = friends
                    campaign['category_url'] = ''
                    campaign['download_time'] = time.time()

                    # db.campaigns.insert_one(campaign)
                    print campaign
                except:
                    import traceback
                    traceback.print_exc()
                    continue
                break
            break
        break

#Conversion of datetime from timestamp
'''import datetime
print(
    datetime.datetime.fromtimestamp(
        int("1474957436")
    ).strftime('%Y-%m-%d %H:%M:%S')
)'''




#category 1: 1000
#category 2: 220
#category 3: 59
#category 4: 56
#category 5: 56
#category 18: 24
#lst = ['California', 'Texas', 'Virginia', 'Seattle', 'help' \
#lst2 = ['Disease', 'Earthquake', 'Cancer']: Sep 26, 2016
#'help' is too common, do not put there again
#['Boston', 'Utah','Florida','Nepal','New York', 'Ohio','Mississippi']
