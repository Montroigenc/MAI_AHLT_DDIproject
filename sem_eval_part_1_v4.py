import re
import collections
from drug_entity_chunker_v2 import ScikitLearnChunker
from nltk import download
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from parser_utils_v2 import parse_drug_iob, pos_tagger, parse_drug_iob_xy
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
import numpy as np
# import sklearn_crfsuite
# from sklearn_crfsuite import scorers
# from sklearn_crfsuite import metrics
from nltk import tree2conlltags
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))

# NLTK dependencies
download('punkt')
download('averaged_perceptron_tagger')

# Directories to find MedLine and DrugBank train and test data
TRAIN_DIR = './Train/DrugBank'
TEST_DIR = './Test/Test for DrugNER task/DrugBank'

# Features to consider when chunking
stemmer = SnowballStemmer('english')

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

def features(tokens, index, history):
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

def to_dataset(parsed_sentences, feature_detector):
    """
    Transform a list of tagged sentences into a scikit-learn compatible POS dataset
    :param parsed_sentences:
    :param feature_detector:
    :return:
    """
    X, y = [], []
    for parsed in parsed_sentences:
        iob_tagged = tree2conlltags(parsed)
        words, tags, iob_tags = zip(*iob_tagged)

        tagged = list(zip(words, tags))

        for index in range(len(iob_tagged)):
            X.append(feature_detector(tagged, index, history=iob_tags[:index]))
            y.append(iob_tags[index])

    return X, y


print("Loading data")
tags = ['drug', 'brand', 'group', 'drug_n']

x_train, y_train, _, _ = parse_drug_iob_xy(TRAIN_DIR, features, tags=tags)
x_test, y_test, sId_test, words_start = parse_drug_iob_xy(TEST_DIR, features, tags=tags)


print("Adapting data format")
# print("train instances: ", len(x_train), "test instances: ", len(x_test))

vectorizer = DictVectorizer(sparse=True)
# x_train_vec = vectorizer.fit_transform(x_train)
x_train_vec = vectorizer.fit(x_train)
x_test_vec = vectorizer.transform(x_test)

classes = ['O', 'B-drug', 'I-drug', 'B-brand', 'I-brand', 'B-group', 'I-group', 'B-drug_n', 'I-drug_n']

# y_train_vec = np.ones(len(y_train)) * -1
# for i, label in enumerate(classes):
#     y_train_vec[np.asarray(y_train) == label] = i

y_test_vec = np.ones(len(y_test)) * -1
for i, label in enumerate(classes):
    y_test_vec[np.asarray(y_test) == label] = i

print("Data loaded")

# # Define classifier models
# clf = Perceptron(verbose=0, n_jobs=-1, max_iter=1)
#
# print("Training...")
# clf.fit(x_train_vec, y_train_vec)
#
# joblib.dump(clf, 'trained_perceptron_classifier.pkl')

clf = joblib.load('trained_perceptron_classifier.pkl')


# print("Testing...")
# y_pred = clf.predict(x_test_vec)
# scores = cross_val_score(clf, x_test_vec, y_test_vec, cv=5)

# print(classification_report(y_test_vec, y_pred, target_names=classes))
# print("CV scores: ", scores, ", mean: ", np.mean(scores), "\n")

print("Writing output TEST file")
words_chain = []
with open("semEval_task1_output.txt", 'w') as outfile:
    for sId, w_start, x in zip(sId_test, words_start, x_test_vec):
        
        prediction = clf.predict(x)
        prediction = classes[int(prediction[0])]
        
        if 'B' in prediction or 'I' in prediction:
            
            entity = prediction.replace('B-', '').replace('I-', '')
            word_data = vectorizer.inverse_transform(x)[0]
            
            for key in word_data:
                if 'word' in str(key) and not('next' in str(key)) and not('prev' in str(key)):
                    word = str(key).replace('word=', ''); break
            
            if 'B' in prediction:
                if len(words_chain) > 0:
                    outfile.write("{}|{}-{}|{}|{}\n".format(words_chain[0], words_chain[1], words_chain[2], words_chain[3], words_chain[4]))
                    words_chain = []
                  
                words_chain.append(sId)
                words_chain.append(w_start)
                words_chain.append(w_start + len(word))
                words_chain.append(word)
                words_chain.append(entity)
                
            else:
                if len(words_chain) > 0:
                    if words_chain[0] == sId:

                        words_chain[2] = words_chain[2] + len(word) + 1  # +1 for the space in between
                        words_chain[3] = '{} {}'.format(words_chain[3], word)
                        
                    else:  # if by mistake a 'I-{*}' arises and we had accumulated 'B-{*}' or 'I-{*}' from last sentence
                        outfile.write("{}|{}-{}|{}|{}\n".format(words_chain[0], words_chain[1], words_chain[2], words_chain[3], words_chain[4]))
                        words_chain = []
                        # words_chain = [sId, w_start, len(word), word, entity]
                        
                        words_chain.append(sId)
                        words_chain.append(w_start)
                        words_chain.append(w_start + len(word))
                        words_chain.append(word)
                        words_chain.append(entity)
                      
                else:  # if by mistake a 'I-{*}' arises but there is no previous 'B-{*}'

                    words_chain.append(sId)
                    words_chain.append(w_start)
                    words_chain.append(w_start + len(word))
                    words_chain.append(word)
                    words_chain.append(entity)
                    
        else:
            if len(words_chain) > 0:
                outfile.write("{}|{}-{}|{}|{}\n".format(words_chain[0], words_chain[1], words_chain[2], words_chain[3], words_chain[4]))
              
            words_chain = []