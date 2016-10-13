from string import punctuation
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import sys

snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

# Tokenize
def tokenize(doc):
    '''
    INPUT: string
    OUTPUT: list of strings
    list of tokenized strings
    '''
    return [wordnet.lemmatize(word) for word in word_tokenize(doc.lower())
            if word not in punctuation and not word.isdigit()]


if __name__ == '__main__':
    doc = sys.argv[1]
    print tokenize(doc)
