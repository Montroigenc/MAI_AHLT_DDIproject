import os

import xmltodict


def parse(doc):
    sentences = extract_sentences(doc)

    return [[extract_text(sentence), parse_sentence(sentence)] for sentence in sentences]


def extract_sentences(doc):
    while not ('sentence' in doc):
        doc = doc['document']

    if '@text' in doc['sentence']:
        return [doc['sentence']]
    else:
        return doc['sentence']


def parse_sentence(sentence):
    parsed = ""
    text = extract_text(sentence)
    entity = extract_entity(sentence)

    if entity is None:
        return parsed

    drug_positions = extract_drug_position(entity)
    bio_array = []; currentPos = 0
    for word in text.split(' '):
        wordBio, currentPos = build_bio(word, text, drug_positions, currentPos)
        bio_array.append(wordBio)

    return parsed.join(bio_array)


def build_bio(word, text, drug_positions, lastPos):
    word_start = text.find(word, lastPos)

    for position in drug_positions:
        drug_start, drug_end = position[0], position[1]

        if word_start == drug_start:
            return "B", word_start

        if drug_start < word_start < drug_end:
            return "I", word_start

    return "O", word_start


def extract_text(sentence):
    return sentence['@text'].rstrip()


def extract_entity(sentence):
    if 'entity' not in sentence:
        return None

    return sentence['entity']


def extract_drug_position(entities):
    if type(entities) != list:
        entities = [entities]

    positions = []
    for entity in entities:
        if "drug" in entity['@type']:
            if ";" in entity['@charOffset']:
                entity = entity['@charOffset'].split(';')
                [(positions.append([int(char) for char in indEntity.split('-')])) for indEntity in entity]
            else:
                positions.append([int(char) for char in entity['@charOffset'].split('-')])
        else:
            positions.append([-1, -1])

    return positions


inputRoot = "/home/aleix/MAI/AHLT/DDI_project/Train"
outputRoot = "/home/aleix/MAI/AHLT/DDI_project/Train/labeledFiles/"

trainDrugBank = os.listdir(inputRoot + "/DrugBank")
trainDrugBank = [inputRoot + "/DrugBank/" + drugFile for drugFile in trainDrugBank]

trainMedLine = os.listdir(inputRoot + "/MedLine")
trainMedLine = [inputRoot + "/MedLine/" + drugFile for drugFile in trainMedLine]

uniqueOutputFile = open(outputRoot + "labelledSentences.txt", "w")
uniqueList = []
for file in trainDrugBank + trainMedLine:

    # file = "/home/aleix/MAI/AHLT/DDI_project/Train/DrugBank/Abarelix_ddi.xml"
    # file = "/home/aleix/MAI/AHLT/DDI_project/Train/DrugBank/L-Carnitine_ddi.xml"
    # file = "/home/aleix/MAI/AHLT/DDI_project/Train/DrugBank/Fenofibrate_ddi.xml"

    inputFile = open(file, 'r')
    dict = xmltodict.parse(inputFile.read())
    inputFile.close()
    outList = parse(dict)

    outputFile = open(outputRoot + (file + ".txt").replace(inputRoot, '').replace('/', '').replace('.xml', ''), "w")

    # put all labels of each xml in one txt file
    [[outputFile.write("%s\n" % interItem) for interItem in outerItem] for outerItem in outList]

    uniqueList.append(outList)

    outputFile.close()

# put all labels in one unique text file
[[[uniqueOutputFile.write("%s\n" % interItem) for interItem in outerItem] for outerItem in list] for list in uniqueList]
uniqueOutputFile.close()
# pass