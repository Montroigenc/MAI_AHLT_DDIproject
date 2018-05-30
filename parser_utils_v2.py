import re
import os
from nltk import word_tokenize, pos_tag
from nltk.chunk import conlltags2tree


def sentence_from_doc(doc):
    doc = doc.replace('\n', '').replace('</document>', '')
    sentences = doc.split('<sentence')[1:]
    return ['<sentence' + str for str in sentences]


def text_from_sentence(sentence):
    regex = r'<sentence .*?text="(.*?)".*?>'
    return re.findall(regex, sentence)[0]

def id_from_sentence(sentence):
    regex = r'<sentence .*?id="(.*?)".*?>'
    return re.findall(regex, sentence)[0]


def drug_pos_from_sentence(sentence, allowed_tags=None):
    regex = r'<entity.*?charOffset="(.*?)".*?type="(.*?)"'
    matches = re.findall(regex, sentence)
    drug_pos = []

    for drug_pos_str, drug_type in matches:

        if allowed_tags and drug_type not in allowed_tags:
            continue

        drug_pos_arr = drug_pos_str.split('-')  # expect '100-121' or '100-110;123-130'
        drug_start, drug_end = int(drug_pos_arr[0]), int(drug_pos_arr[-1]) + 1
        drug_pos.append((drug_start, drug_end, drug_type))

    return drug_pos


def is_drug_bank_filename(path):
    return path.count('DrugBank') > 0 and path[-4:] == '.xml'


def is_med_line_filename(path):
    return path.count('MedLine') > 0 and path[-4:] == '.xml'


def pos_bio_tagger(text, drug_positions):
    text = text.replace('-', ' ')
    pos_tags = pos_tag(word_tokenize(text))
    extended_pos = []
    find_offset = 0

    for word, pos in pos_tags:
        bio = "O"
        word_start = text.find(word, find_offset)
        find_offset += text.count(' ', find_offset, find_offset + len(word)) + len(word)

        for drug_start, drug_end, drug_type in drug_positions:

            if word_start == drug_start:
                bio = f'B-{drug_type}'

            if drug_start < word_start < drug_end:
                bio = f'I-{drug_type}'

        extended_pos.append((word, pos, bio, word_start))

    return extended_pos

def pos_tagger(text):
    text = text.replace('-', ' ')
    pos_tags = pos_tag(word_tokenize(text))

    return pos_tags


def parse_drug_iob(xml_dir, tags=None):
    for root, dirs, files in os.walk(xml_dir):

        for file_name in files:
            path = os.path.join(root, file_name)

            if is_drug_bank_filename(path) or is_med_line_filename(path):
                doc = open(path, 'rb').read().decode('utf-8')
                sentences = sentence_from_doc(doc)

                for sentence in sentences:
                    text = text_from_sentence(sentence)
                    drug_positions = drug_pos_from_sentence(sentence, allowed_tags=tags)
                    tagged_text = pos_bio_tagger(text, drug_positions)

                    if len(tagged_text) > 0:
                        yield conlltags2tree(tagged_text)
              
def parse_drug_iob_xy(xml_dir, feature_detector, tags=None):
    X, y, sId, wSt = [], [], [], []
    for root, dirs, files in os.walk(xml_dir):

        for file_name in files:
            path = os.path.join(root, file_name)

            if is_drug_bank_filename(path) or is_med_line_filename(path):
                doc = open(path, 'rb').read().decode('utf-8')
                sentences = sentence_from_doc(doc)

                for sentence in sentences:
                    id = id_from_sentence(sentence)
                    text = text_from_sentence(sentence)
                    drug_positions = drug_pos_from_sentence(sentence, allowed_tags=tags)
                    tagged_text = pos_bio_tagger(text, drug_positions)

                    if len(tagged_text) > 0:
                        words, word_tags, iob_tags, w_start = zip(*tagged_text)
                        tagged = list(zip(words, word_tags))
                        for index, w_st in zip(range(len(tagged_text)), w_start):
                            sId.append(id)
                            wSt.append(w_st)
                            X.append(feature_detector(tagged, index, history=iob_tags[:index]))
                            y.append(iob_tags[index])
                
    return X, y, sId, wSt