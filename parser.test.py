import unittest
import xmltodict
from parser import *

mock_parse = "OBOOOOOOOOOOOOO"

mock_text = "Serum digoxin levels using an 125I-labelled antigen: Validation of method and observations on cardiac patients."

mock_entity = {
    "@charOffset": "6-12",
    "@id": "DDI-MedLine.d127.s0.e0",
    "@text": "digoxin",
    "@type": "drug"
}

mock_sentence = {
    "@id": "DDI-MedLine.d127.s0",
    "@text": "Serum digoxin levels using an 125I-labelled antigen: Validation of method and observations on cardiac patients.\n",
    "entity": mock_entity
}

mock_sentences = [
    mock_sentence,
    {
        "@id": "DDI-MedLine.d127.s1",
        "@text": "1. "
    },
    {
        "@id": "DDI-MedLine.d127.s2",
        "@text": "Determinations of serum digoxin levels utilizing commercially available kits with an 125I-labelled antigen were precise and not materially different from results obtained with a 3H-labelled antigen. ",
        "entity": {
            "@charOffset": "24-30",
            "@id": "DDI-MedLine.d127.s2.e0",
            "@text": "digoxin",
            "@type": "drug"
        }
    },
    {
        "@id": "DDI-MedLine.d127.s3",
        "@text": "2. "
    },
    {
        "@id": "DDI-MedLine.d127.s4",
        "@text": "In order to approximate the steady state level, serum digoxin levels should be drawn either before or at least six hours following the administration of an oral tablet. ",
        "entity": {
            "@charOffset": "54-60",
            "@id": "DDI-MedLine.d127.s4.e0",
            "@text": "digoxin",
            "@type": "drug"
        }
    },
    {
        "@id": "DDI-MedLine.d127.s5",
        "@text": "3. "
    },
    {
        "@id": "DDI-MedLine.d127.s6",
        "@text": "Concomitantly given thiazide diuretics did not interfere with the absorption of a tablet of digoxin. ",
        "entity": [
            {
                "@charOffset": "20-37",
                "@id": "DDI-MedLine.d127.s6.e0",
                "@text": "thiazide diuretics",
                "@type": "group"
            },
            {
                "@charOffset": "92-98",
                "@id": "DDI-MedLine.d127.s6.e1",
                "@text": "digoxin",
                "@type": "drug"
            }
        ],
        "pair": {
            "@ddi": "false",
            "@e1": "DDI-MedLine.d127.s6.e0",
            "@e2": "DDI-MedLine.d127.s6.e1",
            "@id": "DDI-MedLine.d127.s6.p0"
        }
    },
    {
        "@id": "DDI-MedLine.d127.s7",
        "@text": "4. "
    },
    {
        "@id": "DDI-MedLine.d127.s8",
        "@text": "In the digitalized patient, slow alterations in serum levels after oral administration appeared well correlated with, at least, the negative chronotropic effects of the drug. "
    },
    {
        "@id": "DDI-MedLine.d127.s9",
        "@text": "5. "
    },
    {
        "@id": "DDI-MedLine.d127.s10",
        "@text": "Maximal exercise testing, a maneuver often applied to cardiac patients, does not significantly alter the serum digoxin level.",
        "entity": {
            "@charOffset": "111-117",
            "@id": "DDI-MedLine.d127.s10.e0",
            "@text": "digoxin",
            "@type": "drug"
        }
    }
]

mock_xml = {
    "document": {
        "@id": "DDI-MedLine.d127",
        "sentence": mock_sentences
    }
}

class TestParser(unittest.TestCase):
    # medline
    def test_extract_sentences(self):
        self.assertEqual(extract_sentences(mock_xml), mock_sentences)

    def test_extract_text(self):
        self.assertEqual(extract_text(mock_sentence), mock_text)

    def test_extract_entity(self):
        self.assertEqual(extract_entity(mock_sentence), mock_entity)

    def test_extract_drug_position(self):
        self.assertEqual(extract_drug_position(mock_entity), [[6, 12]])

    def test_parse_sentence(self):
        self.assertEqual(parse_sentence(mock_sentence), mock_parse)

    # drugbank
    def test_parse(self):
        with open("./Train/DrugBank/Almotriptan_ddi.xml") as fd:
            doc = xmltodict.parse(fd.read())

        parse(doc)

if __name__ == '__main__':
    unittest.main()