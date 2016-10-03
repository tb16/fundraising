
import datetime
import time

def days():

    time_string = 'October 5, 2015'
    to_time = time.strptime(time_string, "%B %d, %Y") # convert into struct_time
    seconds = time.mktime(to_time) # convert into epoch argument is struct_time
    t = keep[0]['download_time'] # downloadtime epoch
    delta = (datetime.datetime.fromtimestamp(t)-datetime.datetime.fromtimestamp(seconds))
    return delta.days
