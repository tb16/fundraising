import requests
import sys
import json
import re
import time
import multiprocessing
import multiprocessing.pool
import bs4
import pymongo

db = pymongo.MongoClient().gfm_database
# tab = db['campaigns']
_user_agent = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36')

interval = 2.5
lock = multiprocessing.Lock()
last_time = 0


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
        'category': category,
        'title': title,
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



def list_categories():

    soup = download('https://www.gofundme.com/mvc.php?route=search&term=asdf')
    links = soup.select('.n.cat a')
    for link in links:
        yield 'https://www.gofundme.com' + link['href']



if __name__ == '__main__':
    categories = list(list_categories())
    # categories = categories[1:]
    # print categories
    for j, category_url in enumerate(categories):

        for i in xrange(1,51):
            print {'category': categories[j], 'loop': i }
            soup = download(category_url, params={'page': i})
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
                    campaign['category_url'] = category_url
                    campaign['download_time'] = time.time()

                    db.campaigns.insert_one(campaign)
                    # print campaign
                except:
                    import traceback
                    traceback.print_exc()
                    continue
#category 1: 1000
#category 2: 220
#category 3: 59
#category 4: 56
#category 5: 56
#category 18: 24
#Sep 26: category 7 done
