import itertools
from nltk import tree2conlltags
from nltk.chunk import ChunkParserI
#from sklearn.linear_model import Perceptron
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from nltk.chunk import conlltags2tree
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

class ScikitLearnChunker(ChunkParserI):

    @classmethod
    def to_dataset(cls, parsed_sentences, feature_detector):
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

    @classmethod
    def get_minibatch(cls, parsed_sentences, feature_detector, batch_size=500):
        batch = list(itertools.islice(parsed_sentences, batch_size))
        X, y = cls.to_dataset(batch, feature_detector)
        return X, y

    @classmethod
    def train(cls, x_train, y_train, feature_detector, clf, vectorizer, **kwargs):
        clf.fit(x_train, y_train)

        clf = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', clf)
        ])

        return cls(clf, feature_detector)

    def __init__(self, classifier, feature_detector):
        self._classifier = classifier
        self._feature_detector = feature_detector

    def parse(self, tokens):
        """
        Chunk a tagged sentence
        :param tokens: List of words [(w1, t1), (w2, t2), ...]
        :return: chunked sentence: nltk.Tree
        """
        history = []
        iob_tagged_tokens = []
        for index, (word, tag) in enumerate(tokens):
            iob_tag = self._classifier.predict([self._feature_detector(tokens, index, history)])[0]
            history.append(iob_tag)
            iob_tagged_tokens.append((word, tag, iob_tag))

        return conlltags2tree(iob_tagged_tokens)
    
    # def parsed_sentences2xy_data(self, parsed_sentences):
    #     return self.__class__.to_dataset(parsed_sentences, self._feature_detector)
    
    def score(self, X_test, y_test, classes, cv, **kwargs):
    # def score(self, parsed_sentences, classes, cv):
        """
        Compute the accuracy of the tagger for a list of test sentences
        :param parsed_sentences: List of parsed sentences: nltk.Tree
        :return: float 0.0 - 1.0
        """
        # X_test, y_test = self.__class__.to_dataset(parsed_sentences, self._feature_detector)
        y_pred = self._classifier.predict(X_test)
        if cv > 0:
            scores = cross_val_score(self._classifier, X_test, y_test, cv=5)
        else:
            scores = -1

        return classification_report(kwargs.get('y_test_vec'), y_pred, target_names=classes), scores

    # def score(self, parsed_sentences, classes):
    #     """
    #     Compute the accuracy of the tagger for a list of test sentences
    #     :param parsed_sentences: List of parsed sentences: nltk.Tree
    #     :return: float 0.0 - 1.0
    #     """
    #     # X_test, y_test = self.__class__.to_dataset(parsed_sentences, self._feature_detector)
    #     y_pred = self._classifier.predict(X_test)
    #     scores = cross_val_score(self._classifier, X_test, y_test, cv=5)
		#
    #     return classification_report(y_test, y_pred, target_names=classes), scores

