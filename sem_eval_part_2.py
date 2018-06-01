# DEPENDENCIES Python3
import os
import re
import pickle
import html
from nltk import bigrams, trigrams
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import metrics
from geniatagger import GeniaTaggerClient

# CONSTANTS
# The directories where the gold standard xml is stored (update as required)
TRAIN_DIR = './Train/'
TEST_DIR = './Test/'
GENIA_TAGGER_PATH = '/Users/macbook13/Desktop/geniatagger-3.0.2/geniatagger'
SENT_LENGTH_LIMIT = 500

# These tokens must not clash with the text in the XML!
DRUG_X_TOKEN = '!'
DRUG_Y_TOKEN = '¡'
DRUG_N_TOKEN = '¿'
DIGIT_TOKEN = '?'

# FEATURE BUILDING FUNCTIONS
tagger = GeniaTaggerClient(address='localhost', port=9595)
tokenizer = WordPunctTokenizer()


# PARSING UTILITIES
def is_drug_bank_filename(path):
    return path.count('DrugBank') > 0 and path[-4:] == '.xml'


def is_med_line_filename(path):
    return path.count('MedLine') > 0 and path[-4:] == '.xml'


def sentence_from_doc(doc):
    doc = doc.replace('\n', '').replace('</document>', '')
    sentences = doc.split('<sentence')[1:]
    return ['<sentence' + str for str in sentences]


def text_from_sentence(sentence):
    regex = r'<sentence .*?text="(.*?)".*?>'
    return html.unescape(re.findall(regex, sentence)[0])


def drug_pos_from_sentence(sentence):
    regex = r'<entity.*?id="(.*?)".*?charOffset="(.*?)".*?type="(.*?)".*?text="(.*?)"'
    return re.findall(regex, sentence)


def drug_pairs_from_sentence(sentence):
    regex = r'<pair.*?e1="(.*?)".*?e2="(.*?)".*?(?:ddi="(.*?)"\/>|ddi=".*?" type=".*?"\/>)'
    return re.findall(regex, sentence)


def parse_drug_ddi(xml_dir, labels):

    for root, dirs, files in os.walk(xml_dir):

        for file_name in files:
            path = os.path.join(root, file_name)

            if is_drug_bank_filename(path) or is_med_line_filename(path):
                doc = open(path, 'rb').read().decode('utf-8')
                sentences = sentence_from_doc(doc)

                for sentence in sentences:
                    text = text_from_sentence(sentence)

                    if len(text) > SENT_LENGTH_LIMIT:
                        print('Text too long')
                        continue

                    # Then for each pair we update the token to be either 'DX' or 'DY' depending on the pair
                    for e1, e2, ddi_type in drug_pairs_from_sentence(sentence):
                        sub_text = text

                        if ddi_type != 'false':
                            ddi_type = ddi_type.split('type="')[-1]

                        for drug_id, drug_pos_str, drug_type, drug_text in drug_pos_from_sentence(sentence):
                            replace_token = DRUG_N_TOKEN

                            if drug_id == e1:
                                replace_token = DRUG_X_TOKEN

                            if drug_id == e2:
                                replace_token = DRUG_Y_TOKEN

                            for pos_str in drug_pos_str.split(';'):
                                # print('## DRUG \n', drug_text, '\n', drug_id, '\n', e1, '\n', e2, '\n', pos_str, '\n', sub_text)
                                start, end = pos_str.split('-')
                                start, end = int(start), int(end)+1
                                sub_text = sub_text[0:start] + replace_token * (end - start) + sub_text[end:]

                        assert len(sub_text) == len(text)
                        assert ddi_type in labels

                        yield sub_text, ddi_type


def tokenize(text):
    text = text.replace(',', ' ,').replace('-', ' ').replace('/', ' / ')
    text = re.sub('\d', DIGIT_TOKEN, text)
    text = tokenizer.tokenize(text)
    return ' '.join(text)


def feature_builder(data_reader, classes):
    f_count = 0

    for text, label in data_reader:
        f_count += 1
        print(f_count, label)

        feature_dict = {}
        sent = tokenize(text)
        label = classes[label]
        split_text = sent.split()
        e1_split = [split_text.index(el) for el in split_text if DRUG_X_TOKEN in el][0]
        e2_split = [split_text.index(el) for el in split_text if DRUG_Y_TOKEN in el][0]
        print('Genia tagging')

        try:
            _, _, _, chunk, _ = list(zip(*tagger.parse(sent)))
        except Exception as e:
            print(e)
            continue
        print('Building features')

        # F1 : any word between relation arguments
        for k, i in enumerate(range(e1_split+1, e2_split)):
            feature_dict['F1_'+str(k)] = split_text[i]

        # F2 : any pos between relation arguments
        for k, i in enumerate(range(e1_split+1, e2_split)):
            feature_dict['F2_'+str(k)] = split_text[i]

        # F3 : any bigram between relation arguments
        sent_bigrams = list(bigrams(split_text[e1_split+1:e2_split]))
        for k, bigram in enumerate(sent_bigrams):
            feature_dict['F3_'+str(k)] = '-'.join(bigram)

        # F4 : word preciding first argument
        if e1_split == 0:
            feature_dict['F4'] = '<S>'
        else:
            feature_dict['F4'] = split_text[e1_split - 1]

        # F5 : word preceding second arguments
        if e2_split == 0:
            feature_dict['F5'] = '<S>'
        else:
            feature_dict['F5'] = split_text[e2_split - 1]

        # F6 : any three words succeeding the first arguments
        if e1_split <= len(sent) - 3 :
            sent_trigrams = list(trigrams(split_text[e1_split+1:]))
            for k, trigram in enumerate(sent_trigrams):
                feature_dict['F6_'+str(k)] = '-'.join(trigram)
        else:
            feature_dict['F6_0'] = '<E>'

        # F7 : any three succeeding the second arguments
        if e2_split <= len(sent) - 3 :
            sent_trigrams = list(trigrams(split_text[e2_split+1:]))
            for k, trigram in enumerate(sent_trigrams):
                feature_dict['F7_'+str(k)] = '-'.join(trigram)
        else:
            feature_dict['F7_0'] = '<E>'

        # F8 : sequence of chunk type between relation argumemts
        feature_dict['F8'] = '-'.join(chunk[e1_split+1:e2_split])

        # F9 : string of words between relation arguments
        feature_dict['F9'] = '-'.join(split_text[e1_split+1:e2_split])

        # F13 : Distance between two arguments
        feature_dict['F13'] = abs(sent.index(DRUG_X_TOKEN) - sent.index(DRUG_Y_TOKEN))

        # F14 : Presence of puncuation sign between arguments
        if split_text[e1_split : e2_split] == [','] or ['and'] or ['or'] or ['/']:
            feature_dict['F14'] = True
        else:
            feature_dict['F14'] = False

        yield feature_dict, label


classes = {'false': 0, 'mechanism': 1, 'effect': 2, 'advise': 3, 'int': 4, 'true': 5}

if os.path.isfile('test_features.pickle'):
    test_features = pickle.load(open('test_features.pickle', 'rb'))
else:
    print('Building features, may take a while')
    test_data = parse_drug_ddi(TEST_DIR, classes)
    test_features = list(feature_builder(test_data, classes))
    pickle.dump(test_features, open('test_features.pickle', 'wb'))

X_test, y_test = zip(*test_features)


if os.path.isfile('train_features.pickle'):
    train_features = pickle.load(open('train_features.pickle', 'rb'))
else:
    print('Building features from xml, may take a while')
    train_data = parse_drug_ddi(TRAIN_DIR, classes)
    train_features = list(feature_builder(train_data, classes))
    pickle.dump(train_features, open('train_features.pickle', 'wb'))

X_train, y_train = zip(*train_features)


print("VECTORIZING FEATURES")
vec = DictVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# EVALUATING THE CLASSIFIER
print("FITTING CLASSIFIER")
clf = svm.SVC(kernel='linear', C=0.1).fit(X_train, y_train)
a = clf.score(X_test, y_test)

print("Accuracy", a)
y_pred = clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred,[1,2,3,4],digits=4))