from .utils import get_root
import sqlite3 as sql
import jsonlines
import os
import json
from spacy.lang.en import English
from spacy.lang.de import German
from spacy.lang.es import Spanish

nlps = {
    'en': English,
    'de': German,
    'es': Spanish,
}


class Datasets(object):
    pass


class NQSimplified(Datasets):
    """ Data loader for MLQA dataset
    """

    def __init__(self, dataset, data_path=None):
        """

        """
        root = get_root()
        path = os.path.join(root, "data/nq/simplified")
        if dataset == "valid":
            data_file = os.path.join(path, 'v1.0-simplified-nq-val_f_short_maxlen_5.jsonl')
        elif dataset == "train":
            data_file = os.path.join(path, 'v1.0-simplified-nq-train_f_short_maxlen_5.jsonl')
        elif dataset == "example" or dataset == 'test':
            data_file = os.path.join(path, 'v1.0-simplified-nq-val_f_short_maxlen_5_example.jsonl')
        else:
            raise RuntimeError("Invalid value: dataset \"{}\"".format(dataset))

        if data_path is not None:
            data_file = data_path

        self.reader = jsonlines.open(data_file)

    # def __del__(self):
    #     self.reader.close()

    def get(self):
        for data in self.reader:
            yield data


class MLQADataset(Datasets):
    """ Data loader for MLQA dataset
    """

    def __init__(self, dataset, lang_context, lang_question, data_path=None):
        """ Returns scored documents in multiple languages.

        Parameters:
        dataset (str): ['dev', 'test']
        lang_context (str): context language
        lang_question (str): question language

        Returns:

        [scoreDocs]: ordered list of scored documents by their score

        """
        filename = dataset + '-context-' + lang_context + '-question-' + lang_question + ".json"
        if data_path is None:
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
                        'title': doc['title'],
                        'context': paragraph['context'],
                        'question': qa['question'],
                        'qid': qa['id'],
                        'answer': qa['answers'][0]['text'],
                        'start': qa['answers'][0]['answer_start'],
                    }


class Wiki(Datasets):
    def __init__(self, lang, data_path=None, max_length=1000, paragraph_overlap=False):
        # TODO  paragraph_overlap needs a check
        if data_path is None:
            root = get_root()
            path = os.path.join(root, 'data', 'wiki', 'tmp_' + lang + 'wiki_chenprep.db')
        else:
            path = os.path.join(data_path, lang + 'wiki_chenprep.db')

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
            for sent in doc.sents:
                sentence = sent.string.strip()

                if len(paragraph) + len(sentence) <= self.max_len:
                    if paragraph == "":
                        paragraph += sentence
                    else:
                        paragraph += " " + sentence
                    # overlap paragraphs by one sentence
                else:
                    yield {
                        'title': title,
                        'context': paragraph,
                    }
                    if self.overlap:
                        paragraph = prev_sentence + " " + sentence
                    else:
                        paragraph = sentence

                if self.overlap:
                    prev_sentence = sentence
            # yields the rest in the document
            yield {
                'title': title,
                'context': paragraph,
            }

    def close(self):
        self.conn.close()
