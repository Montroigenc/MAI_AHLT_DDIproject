import re
import os
from nltk import word_tokenize, pos_tag


def sentence_from_doc(doc):
    doc = doc.replace('\n', '').replace('</document>', '')
    sentences = doc.split('<sentence')[1:]
    return ['<sentence' + str for str in sentences]


def text_from_sentence(sentence):
    regex = r'<sentence .*?text="(.*?)".*?>'
    return re.findall(regex, sentence)[0]


def drug_pos_from_sentence(sentence):
    regex = r'<entity.*?charOffset="(.*?)".*?type="(.*?)"'
    return re.findall(regex, sentence)


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

        for drug_pos_str, drug_type in drug_positions:
            drug_pos_arr = drug_pos_str.split('-')  # expect '100-121' or '100-110;123-130'
            drug_start, drug_end = int(drug_pos_arr[0]), int(drug_pos_arr[-1]) + 1

            if word_start == drug_start:
                bio = f'B-{drug_type}'

            if drug_start < word_start < drug_end:
                bio = f'I-{drug_type}'

        extended_pos.append((word, pos, bio))

    return extended_pos


def parse_drug_iob(xml_dir):

    for root, dirs, files in os.walk(xml_dir):

        for file_name in files:
            path = os.path.join(root, file_name)

            if is_drug_bank_filename(path) or is_med_line_filename(path):
                doc = open(path, 'rb').read().decode('utf-8')
                sentences = sentence_from_doc(doc)

                for sentence in sentences:
                    text = text_from_sentence(sentence)
                    drug_positions = drug_pos_from_sentence(sentence)
                    tagged_text = pos_bio_tagger(text, drug_positions)

                    if len(tagged_text) > 0:
                        yield [((word, pos), bio) for word, pos, bio in tagged_text]