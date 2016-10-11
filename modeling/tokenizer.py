from string import punctuation
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer


snowball = SnowballStemmer('english')

# Tokenize
def tokenize(doc):
    '''
    INPUT: string
    OUTPUT: list of strings
    list of tokenized strings
    '''
    return [snowball.stem(word) for word in word_tokenize(doc.lower())
            if word not in punctuation and not word.isdigit()]
