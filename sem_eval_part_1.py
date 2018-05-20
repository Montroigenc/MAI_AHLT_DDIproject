import re
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize, pos_tag, ne_chunk

TRAIN_X_DIR = './Train/DrugBank'
TRAIN_Y_DIR = './Train/DrugBankOutput'

TEST_X_DIR = './Test/DrugBank'
TEST_Y_DIR = './Test/DrugBankOutput'


def extract_sentences(doc):
    regex = r'<sentence.*?text=".*?">'
    return re.findall(regex, doc)


def extract_text(sentence):
    regex = r' text="(.*?)">'
    return re.findall(regex, sentence)


def sentence_text_from_doc(doc):
    regex = r'<sentence.*? text="(.*?)">'
    return re.findall(regex, doc)


def parse_drug_xml(xml_dir, limit=None):
    reached = 0

    for f in listdir(xml_dir):

        if limit is not None and limit == reached:
            break

        path = join(xml_dir, f)

        if not isfile(path):
            continue

        doc = open(path, 'r').read()
        texts = sentence_text_from_doc(doc)
        ner = [ne_chunk(pos_tag(word_tokenize(text))) for text in texts]
        print(ner)




        reached += 1

    return


print('Evaluating SEM-EVAL part 1...')
parse_drug_xml(TRAIN_X_DIR, 5)
# x_train, y_train, x_test, y_test = parse_med_xml(TRAIN_DIR), parse_med_xml(TEST_DIR)
# classifier = load_classifier()
