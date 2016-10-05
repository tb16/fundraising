import pandas as pd
import numpy as numpy
import re
from string import punctuation

'''
Count word in the story, title.
Count sentence in the story
'''


def count_words(text):

    '''
    Count words
    '''

    return len(unicode(text, errors = 'ignore').split())
    # r = re.compile(r'[{}]'.format(punctuation))
    # new_text = r.sub(' ',text)
    # return len(new_text.split())

def count_sentences(text):
    '''
    count sentences
    '''
    return len(unicode(text, errors = 'ignore').split('.'))
