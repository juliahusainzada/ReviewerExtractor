import pandas as pd
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer

def stopword_loader(directorypath):
    '''
    Loads in stop word text file. 
    directorypath = the location and name of the stopword file (string)
    '''
    txt_file = open(directorypath, "r")
    file_content = txt_file.read()

    content_list = file_content.split("\n")
    txt_file.close()

    stop_words = content_list
    
    return stop_words

lemmer = WordNetLemmatizer()
punctuation = [',', ':', ';', '.', "'", '"', '(', ')', '’', 'SUB', 'SUP', 'sub', 'sup', 'l&gt', 'l&lt', 'lt', 'gt', 'ch']

def preprocess_text(abstract, stop_words):
    '''
    Cleans, tokenizes, removes stopwords, and lemmatizes text
    Done once per abstract
    '''

    # Step 1: Remove non-letters
    letters = re.sub("[^a-zA-Z]+", " ", abstract)
    
    # Step 2: Make it all lowercase.
    lower = letters.lower()
    
    # Step 3: Tokenize the abstract.
    tokens = word_tokenize(lower)
    
    # Step 4: Filter stopwords and punctuation
    filtered = [
        w for w in tokens
        if w not in stop_words
        and w not in punctuation
        and w.isalpha()
    ]
    
    # Step 5: Lemm the words.
    lemmed = [lemmer.lemmatize(word) for word in filtered]
    
    return lemmed

# --------------------------------------------------
# Unified N-Gram Generator
# --------------------------------------------------

def compute_top_ngrams(tokens, n=1, top_k=10):
    """
    Computes top n-grams from preprocessed token list.
    """
    if n == 1:
        counts = Counter(tokens)
    else:
        counts = Counter(ngrams(tokens, n))

    return counts.most_common(top_k)

# --------------------------------------------------
# Public Functions (Backward Compatible)
# --------------------------------------------------

def topwords(abstract, directorypath):
    stop_words = stopword_loader(directorypath)
    tokens = preprocess_text(abstract, stop_words)
    return compute_top_ngrams(tokens, n=1)


def topbigrams(abstract, directorypath):
    stop_words = stopword_loader(directorypath)
    tokens = preprocess_text(abstract, stop_words)
    return compute_top_ngrams(tokens, n=2)


def toptrigrams(abstract, directorypath):
    stop_words = stopword_loader(directorypath)
    tokens = preprocess_text(abstract, stop_words)
    return compute_top_ngrams(tokens, n=3)