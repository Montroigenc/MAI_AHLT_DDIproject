import re
from time import time
import collections
from nltk import download
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# Features to consider when chunking
stemmer = SnowballStemmer('english')


def vectorize_attr(original_attr, clas):
    vec_attr = np.ones(len(original_attr)) * -1
    for i, label in enumerate(clas):
        vec_attr[np.asarray(original_attr) == label] = int(i)
    
    return vec_attr

def shape(word):
    word_shape = 'other'
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        word_shape = 'number'
    elif re.match('\W+$', word):
        word_shape = 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        word_shape = 'capitalized'
    elif re.match('[A-Z]+$', word):
        word_shape = 'uppercase'
    elif re.match('[a-z]+$', word):
        word_shape = 'lowercase'
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        word_shape = 'camelcase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        word_shape = 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        word_shape = 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        word_shape = 'contains-hyphen'
    
    return word_shape


def dict_features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
    # Pad the sequence with placeholders
    tokens = [('__START2__', '__START2__'), ('__START1__', '__START1__')] + list(tokens) + [('__END1__', '__END1__'),
                                                                                            ('__END2__', '__END2__')]
    history = ['__START2__', '__START1__'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[-1]
    prevpreviob = history[-2]

    feat_dict = {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'shape': shape(word),

        'next-word': nextword,
        'next-pos': nextpos,
        'next-lemma': stemmer.stem(nextword),
        'next-shape': shape(nextword),

        'next-next-word': nextnextword,
        'next-next-pos': nextnextpos,
        'next-next-lemma': stemmer.stem(nextnextword),
        'next-next-shape': shape(nextnextword),

        'prev-word': prevword,
        'prev-pos': prevpos,
        'prev-lemma': stemmer.stem(prevword),
        'prev-iob': previob,
        'prev-shape': shape(prevword),

        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
        'prev-prev-lemma': stemmer.stem(prevprevword),
        'prev-prev-iob': prevpreviob,
        'prev-prev-shape': shape(prevprevword),
    }

    return feat_dict


def array_features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
    # Pad the sequence with placeholders
    tokens = [('__START2__', '__START2__'), ('__START1__', '__START1__')] + list(tokens) + [('__END1__', '__END1__'),
                                                                                            ('__END2__', '__END2__')]
    history = ['__START2__', '__START1__'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[-1]
    prevpreviob = history[-2]

    feat = [
        'word='+ word,
        'lemma='+ stemmer.stem(word),
        'pos='+ pos,
        'shape='+ shape(word),

        'next-word='+ nextword,
        'next-pos='+ nextpos,
        'next-lemma='+ stemmer.stem(nextword),
        'next-shape='+ shape(nextword),

        'next-next-word='+ nextnextword,
        'next-next-pos='+ nextnextpos,
        'next-next-lemma='+ stemmer.stem(nextnextword),
        'next-next-shape='+ shape(nextnextword),

        'prev-word='+ prevword,
        'prev-pos='+ prevpos,
        'prev-lemma='+ stemmer.stem(prevword),
        'prev-iob='+ previob,
        'prev-shape='+ shape(prevword),

        'prev-prev-word='+ prevprevword,
        'prev-prev-pos='+ prevprevpos,
        'prev-prev-lemma='+ stemmer.stem(prevprevword),
        'prev-prev-iob='+ prevpreviob,
        'prev-prev-shape='+ shape(prevprevword),
    ]

    return feat

def array_features_v2(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
    # Pad the sequence with placeholders
    tokens = [('__START2__', '__START2__'), ('__START1__', '__START1__')] + list(tokens) + [('__END1__', '__END1__'),
                                                                                            ('__END2__', '__END2__')]
    history = ['__START2__', '__START1__'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[-1]
    prevpreviob = history[-2]

    feat = [
        'word='+ word,
        'lemma='+ stemmer.stem(word),
        'pos='+ pos,
        'shape='+ shape(word),

        'next-word='+ nextword,
        'next-pos='+ nextpos,
        'next-lemma='+ stemmer.stem(nextword),
        'next-shape='+ shape(nextword),

        'next-next-word='+ nextnextword,
        'next-next-pos='+ nextnextpos,
        'next-next-lemma='+ stemmer.stem(nextnextword),
        'next-next-shape='+ shape(nextnextword),

        'prev-word='+ prevword,
        'prev-pos='+ prevpos,
        'prev-lemma='+ stemmer.stem(prevword),
        'prev-iob='+ previob,
        'prev-shape='+ shape(prevword),

        'prev-prev-word='+ prevprevword,
        'prev-prev-pos='+ prevprevpos,
        'prev-prev-lemma='+ stemmer.stem(prevprevword),
        'prev-prev-iob='+ prevpreviob,
        'prev-prev-shape='+ shape(prevprevword),
    ]

    return feat

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))