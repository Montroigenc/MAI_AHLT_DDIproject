import xmltodict
import json

test_xml_path = "./Train/Medline/1109556.xml"

with open(test_xml_path) as fd:
    doc = xmltodict.parse(fd.read())
    print(json.dumps(doc, sort_keys=True, indent=4))


def parse(doc):
    sentences = extract_sentences(doc)
    return [[extract_text(sentence), parse_sentence(sentence)] for sentence in sentences]


def extract_sentences(doc):
    return doc['document']['sentence']


def parse_sentence(sentence):
    text = extract_text(sentence)
    entity = extract_entity(sentence)
    drug_positions = extract_drug_position(entity)
    bio_array = [build_bio(word, text, drug_positions) for word in text.split(' ')]
    return ''.join(bio_array)


def build_bio(word, text, drug_positions):
    word_start = text.find(word)
    bio = "O"

    for position in drug_positions:
        drug_start, drug_end = position[0], position[1]

        if word_start == drug_start: return "B"

        if drug_start < word_start < drug_end: return "I"

    return bio


def extract_text(sentence):
    return sentence['@text'].rstrip()


def extract_entity(sentence):
    return sentence['entity']


def extract_drug_position(entities):
    if type(entities) != list: entities = [entities]

    return [extract_char_offset_array(entity) for entity in entities]

def extract_char_offset_array(entity):
    if entity['@type'] != "drug": return [-1, -1]

    return [int(char) for char in entity['@charOffset'].split('-')]
