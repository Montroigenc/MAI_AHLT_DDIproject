from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI, conlltags2tree


class DrugEntityChunker(ChunkParserI):
    def __init__(self, train_sents=None, features=None, tagger=None, **kwargs):

        if train_sents is None and tagger is None:
            raise ValueError('No training sentences or tagger provided')

        if tagger:
            self.tagger = tagger
        else:
            self.tagger = ClassifierBasedTagger(train=train_sents, feature_detector=features, **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk. Tree format
        return conlltags2tree(iob_triplets)