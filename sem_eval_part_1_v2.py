import re
from drug_entity_chunker_v2 import ScikitLearnChunker
from nltk import download
from nltk.stem.snowball import SnowballStemmer
from parser_utils import parse_drug_iob
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
import numpy as np

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from nltk import tree2conlltags

# NLTK dependencies
download('punkt')
download('averaged_perceptron_tagger')

# Directories to find MedLine and DrugBank train and test data
TRAIN_DIR = './Train/DrugBank'
TEST_DIR = './Test/Test for DrugNER task/DrugBank'


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


# Features to consider when chunking
stemmer = SnowballStemmer('english')


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


print("Loading train data")
tags = ['drug', 'brand', 'group', 'drug_n']
train_reader = parse_drug_iob(TRAIN_DIR, tags=tags)
test_reader = parse_drug_iob(TEST_DIR, tags=tags)

print("Adapting data format")
x_train, y_train = to_dataset(train_reader, features)
x_test, y_test = to_dataset(test_reader, features)
print("train instances: ", len(x_train), "test instances: ", len(x_test))

vectorizer = DictVectorizer(sparse=True)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

classes = ['O', 'B-drug', 'I-drug', 'B-brand', 'I-brand', 'B-group', 'I-group', 'B-drug_n', 'I-drug_n']

y_train_vec = np.ones(len(y_train)) * -1
for i, label in enumerate(classes):
    y_train_vec[np.asarray(y_train) == label] = i
    
y_train_vec = y_train_vec.tolist()
    
y_test_vec = np.ones(len(y_test)) * -1
for i, label in enumerate(classes):
    y_test_vec[np.asarray(y_test) == label] = i

y_test_vec = y_test_vec.tolist()

# Define classifier models
clf_perceptron = Perceptron(verbose=0, n_jobs=-1, max_iter=1)
clf_tree = DecisionTreeClassifier(criterion='entropy')
clf_NB = MultinomialNB()
clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100), random_state=1)
clf_CRF = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)

clfs = [clf_CRF, clf_tree, clf_NB, clf_MLP]
#clfs = [clf_perceptron, clf_tree, clf_NB, clf_MLP, clf_CRF]

names = ["Conditional random fields", "Decision tree", "Naive Bayes", "Multi Layer Perceptron"]
batch_models = ["Perceptron", "Naive Bayes"]
# names = ["Perceptron", "Decision tree", "Naive Bayes", "Multi Layer Perceptron", "Conditional random fields"]

print("Training chunkers")
for name, clf in zip(names, clfs):
    print("TRAINING ", name)
    # print("X_test: ", len(x_test), "y_test: ", len(y_test))
    # if name in batch_models:
    #     chunker = ScikitLearnChunker.train(train_reader, feature_detector=features, all_classes=classes, clf=clf, vectorizer=vectorizer, batch_size=300)
    # else:
    chunker = ScikitLearnChunker.train(x_train, y_train, feature_detector=features, clf=clf, vectorizer=vectorizer)
    # chunker = ScikitLearnChunker.train(train_reader, feature_detector=features, all_classes=classes, clf=clf, vectorizer=vectorizer, batch_size=0, x_train=x_train_vec, y_train=y_train_vec)
        
    
    print("TESTING ", name)
    accuracy, cv_scores = chunker.score(x_test, y_test, classes, cv=5, x_test_vec=x_test_vec, y_test_vec=y_test_vec)
    print("Accuracies: ", accuracy)
    print("CV scores: ", cv_scores, ", mean: ", np.mean(cv_scores))

# print("Training chunker 1")
# # chunker = ScikitLearnChunker.train(train_reader, feature_detector=features, all_classes=classes, clf=clf1, batch_size=300)
#
# print("Obtaining xy data")
# x_test, y_test = chunker.parsed_sentences2xy_data(test_reader)
# # x_test, y_test = trained_clfs[0].parsed_sentences2xy_data(test_reader)
#
# print("Evaluating chunker 1")
# accuracy = chunker.score(x_test, y_test, classes, cv=5)
# print("Accuracy alone", accuracy)
#
# print("Evaluating chunkers")
# accuracies = []
#
# for i, clf in enumerate(trained_clfs):
#     accuracies.append(clf.score(x_test, y_test, classes, cv=5))
# #
# for i, acc in enumerate(accuracies):
#     print("Accuracies ", acc)
    
    

