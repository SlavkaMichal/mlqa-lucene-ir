import sys, os, lucene, threading, time
from datetime import datetime
from .translator import Translator
from .utils import get_root
from .datasets import MLQADataset, Wiki
from glob import glob
import numpy as np
import pdb
import math
import json

# lucene imports
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.de import GermanAnalyzer
from org.apache.lucene.analysis.es import SpanishAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer


analyzers = {
        'standard':StandardAnalyzer,
        'en':EnglishAnalyzer,
        'es':SpanishAnalyzer,
        'de':GermanAnalyzer
        }

class Retriever(object):
    def __init__(self, k1=None, b=None):
        if k1 == None:
            self.k1=1.8
        else:
            self.k1=k1

        if b == None:
            self.b=0.1
        else:
            self.b=b

    def dataname(self, dataset, context, question):
        return dataset+'-context-'+context+'-question-'+question

    def get_index(self, lang, dataset, index_path=None, suffix=""):
        if suffix != "":
            idxdir = "{}-{}-{}.index".format(dataset,lang,suffix)
        else:
            idxdir = "{}-{}.index".format(dataset,lang)
        if index_path == None:
            root = get_root()
            return os.path.join(root, 'data', 'indexes', idxdir)
        else:
            return os.path.join(index_path, idxdir)


class Indexer(Retriever):
    def __init__(self, lang, dataset, analyzer, index_path=None, data_path=None, ram_size=2048 ):
        """ Returns scored documents in multiple languages.

        Parameters:
        dataset  (str): ['mlqa_dev', 'mlqa_test', 'wiki']
        lang     (str): ['en', 'es', 'de']
        anlyzer  (str): ['en', 'es', 'de', 'standard']
        ram_size (int): Size of memory used while indexing

        Returns:
        """
        super().__init__()

        idxdir = self.get_index(lang, dataset, index_path)
        self.mlqa = True
        if dataset == 'mlqa_dev':
            self.dataset = MLQADataset('dev', lang, lang, data_path)
        elif dataset == 'mlqa_test':
            self.dataset = MLQADataset('test', lang, lang, data_path)
        elif dataset == 'wiki':
            self.mlqa = False
            self.dataset = Wiki(lang, data_path)
        else:
            raise RuntimeError("No dataloader for {}".format(dataset))

        # stores index files, poor concurency try NIOFSDirectory instead
        store = SimpleFSDirectory(Paths.get(idxdir))
        # limit max. number of tokens per document.
        # analyzer will not consume more tokens than that
        #analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        # configuration for index writer
        config = IndexWriterConfig(analyzers[analyzer]())
        # creates or overwrites index
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        # setting similarity BM25Similarity(k1=1.2,b=0.75)
        similarity = BM25Similarity(self.k1, self.b)
        config.setSimilarity(similarity)
        config.setRAMBufferSizeMB(float(ram_size))
        # create index writer
        self.writer = IndexWriter(store, config)

        self.ftdata = FieldType()
        self.ftmeta = FieldType()
        # IndexSearcher will return value of the field
        self.ftdata.setStored(True)
        self.ftmeta.setStored(True)
        # will be analyzed by Analyzer
        self.ftdata.setTokenized(True)
        self.ftmeta.setTokenized(False)
        # what informations are stored (probabli DOCS would be sufficient)
        # DOCS: Only documents are indexed: term frequencies and positions are omitted.
        #       Phrase and other positional queries on the field will throw an exception,
        #       and scoring will behave as if any term in the document appears only once.
        # DOCS_AND_FREQS: Only documents and term frequencies are indexed: positions are
        #       omitted. This enables normal scoring, except Phrase and other positional
        #       queries will throw an exception.
        # DOCS_AND_FREQS_AND_POSITIONS: Indexes documents, frequencies and positions.
        #       This is a typical default for full-text search: full scoring is enabled
        #       and positional queries are supported.
        self.ftdata.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
        self.ftmeta.setIndexOptions(IndexOptions.DOCS)
        # instantiate some reusable objects
        # TODO: create document, add fields then change only field value and
        # re-add document
        self.doc = Document()
        # Id cannot be reused because there is multiple values
        # I could store list of fields and add one if its not enough
        #self.fieldId = Field("id", "dummy", self.ftmeta)
        self.fieldTitle = Field("title", "dummy", self.ftdata)
        self.doc.add(self.fieldTitle)
        self.fieldContext = Field("context", "dummy", self.ftdata)
        self.doc.add(self.fieldContext)
        self.fieldIds     = [Field("id", "dummy", self.ftmeta)]

    def addDoc(self, ids, title, context):
        # to save resources field objects are not created each time a new
        # document is being added. fieldIds keeps already created objects
        for n, i in enumerate(ids):
            if n < len(self.fieldIds):
                self.fieldIds[n].setStringValue(i)
            else:
                self.fieldIds.append(Field("id", i, self.ftmeta))
            self.doc.add(self.fieldIds[n])

        self.fieldTitle.setStringValue(title)
        self.fieldContext.setStringValue(context)
        self.writer.addDocument(self.doc)
        # because the number of ids is not known, they have to be deleted
        # otherwise there could contain values from previous iteration
        self.doc.removeFields("id")

    def createIndex(self):
        ids = []
        for i, doc in enumerate(self.dataset.get()):
            if self.mlqa:
                ids = doc['qid']
            self.addDoc(ids, doc['title'], doc['context'])
        self.commit()

    def commit(self):
        self.writer.commit()
        self.writer.close()
        if not self.mlqa:
            self.dataset.close()


class Searcher(Retriever):
    def __init__(self, lang=None, dataset=None, analyzer=None, index_path=None, k1=None, b=None):
        super().__init__(k1, b)
        print("Searcher k1: {}, b: {}", self.k1, self.b)
        self.similarity = BM25Similarity(self.k1, self.b)
        self.searcher = {}
        self.parser = {}
        self.languages = []
        self.lang = lang
        self.dataset = dataset
        self.__call__ = self.query
        if lang != None or dataset != None or analyzer != None:
            self.addLang(lang, dataset, analyzer, index_path)

    def addLang(self, lang, dataset, analyzer, index_path=None):
        self.languages.append(lang)
        idxdir = self.get_index(lang, dataset, index_path)
        directory = SimpleFSDirectory(Paths.get(idxdir))
        self.searcher[lang] = IndexSearcher(DirectoryReader.open(directory))
        self.parser[lang]   = QueryParser("context", analyzers[analyzer]())
        self.searcher[lang].setSimilarity(self.similarity)
        self.lang = lang

    def printResult(self, scoreDocs):
        print("Number of retrieved documents:", len(scoreDocs))
        for scoreDoc in scoreDocs:
            doc = self.searcher[self.lang].doc(scoreDoc.doc)
            print("Score:", scoreDoc.score)
            self.printDoc(doc)

    def printDoc(self, doc):
            ids = doc.getFields('id')
            for id in ids:
                print("Id:", id.stringValue())
            print("Name:", doc.get("title").encode('utf-8'))
            print("Context:", doc.get("context").encode('utf-8'))
            print("------------------------------------------------------")

    def getDoc(self, scoreDoc):
        return self.searcher[self.lang].doc(scoreDoc.doc)

    def queryTest(self, command):
        q = self.query(command, self.lang, 5)
        self.printResult(q)
        return q

    def query(self, command, lang=None, n=50):
        """
        Retrieve documents for question
        """
        if lang != None:
            self.lang = lang
        if self.lang not in self.languages:
            raise RuntimeError("Language '{}' not added".format(lang))

        esccommand = self.parser[self.lang].escape(command)
        query = self.parser[self.lang].parse(esccommand)
        scoreDocs = self.searcher[self.lang].search(query, n).scoreDocs
        return scoreDocs

    #def queryMulti(self, command, lang, n=50, p=1):
    #    """ Returns scored documents in multiple languages.

    #    Parameters:
    #    command (str): query string
    #    lang    (str): language in which is the query
    #    n       (int): number of documents retrieved
    #    p       (float): reduces number of retrieved documents from each language
    #                     e.g.: for 3 languages, n = 50 and p = 0.5 from each language
    #                     25 documents will be retrieved.
    #                     Must satisfy n*len(langs)*p >= n

    #    Returns:

    #    [scoreDocs]: ordered list of scored documents by their score

    #    """
    #    scoreDocs = []
    #    for to in self.languages:
    #        transl_comm = self.translator(lang, to, command)
    #        scoreDocs.append(self.query(transl_comm, to, int(n*p)))
    #    return scoreDocs.sort(key=lambda x: x.score, reverse=True)[:n]

