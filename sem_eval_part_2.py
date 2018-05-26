import re
from drug_entity_chunker import ScikitLearnChunker
from nltk import download
from nltk.stem.snowball import SnowballStemmer
from parser_utils import parse_drug_iob

# NLTK dependencies
download('punkt')
download('averaged_perceptron_tagger')

# Directories to find MedLine and DrugBank train and test data
TRAIN_DIR = './Train/DrugBank'
TEST_DIR = './Test/Test for DrugNER task/DrugBank'

print("Loading train data")
tags = ['drug']
# train_reader = parse_drug_iob(TRAIN_DIR, tags=tags)
test_reader = parse_drug_iob(TEST_DIR, tags=tags)