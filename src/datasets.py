from .utils import get_root
import sqlite3 as sql
import os
import json
from spacy.lang.en import English
from spacy.lang.de import German
from spacy.lang.es import Spanish

nlps = {
        'en':English,
        'de':German,
        'es':Spanish,
        }
class Datasets(object):
    pass

class MLQADataset(Datasets):
    """ Data loader for MLQA dataset
    """
    def __init__(self, dataset, langC, langQ, data_path=None):
        """ Returns scored documents in multiple languages.

        Parameters:
        dataset (str): ['dev', 'test']
        langContext (str): context language
        langQuestion (str): question language

        Returns:

        [scoreDocs]: ordered list of scored documents by their score

        """
        filename = dataset+'-context-'+langC+'-question-'+langQ+".json"
        if data_path == None:
            root = get_root()
            path = os.path.join(root, 'data', 'MLQA_V1', dataset, filename)
        else:
            path = os.path.join(data_path, filename)
        print("Mlqa dataset from: ", path)
        with open(path) as fp:
            self.data = json.load(fp)['data']
        return

    def get(self):
        for doc in self.data:
            for paragraph in doc['paragraphs']:
                for qa in paragraph['qas']:
                    yield {
                        'title':doc['title'],
                        'context':paragraph['context'],
                        'question':qa['question'],
                        'qid':qa['id'],
                        'answer':qa['answers'][0]['text'],
                        'start':qa['answers'][0]['answer_start'],
                        }

class Wiki(Datasets):
    def __init__(self, lang, data_path=None, max_length=1000, paragraph_overlap=False):
        # TODO  paragraph_overlap needs a check
        if data_path == None:
            root = get_root()
            path = os.path.join(root, 'data', 'wiki', 'tmp_'+lang+'wiki_chenprep.db')
        else:
            path = os.path.join(data_path, lang+'wiki_chenprep.db')

        print("Path to db: ", path)
        self.conn = sql.connect(path)
        self.c = self.conn.cursor()
        self.nlp = nlps[lang]()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.max_len = max_length
        self.overlap = paragraph_overlap

    def get(self):
        for (title, raw) in self.c.execute('SELECT * FROM documents'):
            doc = self.nlp(raw)
            paragraph = ""
            prev_sentence = ""
            sentence = ""
            for sent in doc.sents:
                sentence = sent.string.strip()

                if len(paragraph) + len(sentence) <= self.max_len:
                    if paragraph == "":
                        paragraph += sentence
                    else:
                        paragraph += " "+sentence
                    # overlap paragraphs by one sentence
                else:
                    yield {
                        'title':title,
                        'context':paragraph,
                        }
                    if self.overlap:
                        paragraph = prev_sentence+" "+sentence
                    else:
                        paragraph = sentence

                if self.overlap:
                    prev_sentence = sentence
            # yeild the rest in the document
            yield {
                'title':title,
                'context':paragraph,
                }

    def close(self):
        self.conn.close()

