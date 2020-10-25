import sys, os, lucene, threading, time
from datetime import datetime
from retrieval.src.translator import Translator
from retrieval.src.utils import get_root, load_data
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

class Retriever(object):
    def __init__(self):
        self.k1=1.8
        self.b=0.1
    def dataname(self, dataset, context, question):
        return dataset+'-context-'+context+'-question-'+question


class Indexer(Retriever):
    def __init__(self, storedir, analyzer, datadir=None, ram_size=2048 ):
        if not os.path.exists(storedir):
            os.mkdir(storedir)
        # stores index files, poor concurency try NIOFSDirectory instead
        store = SimpleFSDirectory(Paths.get(storedir))
        # limit max. number of tokens per document.
        # analyzer will not consume more tokens than that
        #analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        # configuration for index writer
        config = IndexWriterConfig(analyzer)
        # creates or overwrites index
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        # setting similarity BM25Similarity(k1=1.2,b=0.75)
        similarity = BM25Similarity(self.k1, self.b)
        config.setSimilarity(similarity)
        config.setRAMBufferSize(ram_size)
        # create index writer
        self.writer = IndexWriter(store, config)

        self.ftdata = FieldType()
        self.ftmeta = FieldType()
        # IndexSearcher will return value of the field
        self.ftdata.setStored(False)
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
        self.fieldDocName = Field("docname", "dummy", self.ftdata)
        self.doc.add(fieldDocName)
        self.fieldContext = Field("context", "dummy", self.ftdata)
        self.doc.add(fieldContext)


        if datadir != None:
            print("Indexing files")
            indexDocs(datadir, writer)
            writer.commit()
            writer.close()
            print("done")

    def addDoc(self, ids, title, context):
        print("adding:", title.encode('utf-8'))
        #doc = Document()
        for i in ids:
            doc.add(Field("id", i, self.ftmeta))
        fieldDocName.setStringValue(title)
        fieldContext.setStringValue(context)
        #doc.add(Field("docname", title, self.ftdata))
        #doc.add(Field("context", context, self.ftdata))
        self.writer.addDocument(doc)
        self.doc.removeFields("id")

    def indexDocs(self, datadir):
        """ Index documents in separate files """
        #for datadir, dirnames, filenames in os.walk(root):
        for filename in glob(datadir):
            if not filename.endswith('.txt'):
                continue
            path = os.path.join(datadir, filename)
            with open(path, 'r', encoding='utf-8') as fp:
                contents = fp.read().splitlines()
            self.addDoc(path, contents[0], contents[1])

    def createIndex(self, data, dataset, lang):
        dataname = self.dataname(dataset, lang, lang)
        data = data['mlqa_'+dataset][dataname]['data']
        for doc in data:
            title = doc['title']
            for paragraph in doc['paragraphs']:
                ids = []
                for qa in paragraph['qas']:
                    ids.append(qa['id'])
                self.addDoc(ids, title, paragraph['context'])
        self.commit()

    def commit(self):
        self.writer.commit()
        self.writer.close()

class Searcher(Retriever):
    def __init__(self, lang=None, index=None, analyzer=None):
        if type(analyzer) != type(index):
            raise RuntimeError("For multilingual search you need"+
                    " multiple analyzers and indexes!")
        self.similarity = BM25Similarity(self.k1, self.b)
        self.searcher = {}
        self.parser = {}
        self.languages = []
        self.translator = Translator([])
        if lang != None or index != None or anlyzer != None:
            self.addLang(lang, index, analyzer)

    def addLang(self, lang, index, analyzer):
        self.languages.append(lang)
        directory = SimpleFSDirectory(Paths.get(index))
        self.searcher[lang] = IndexSearcher(DirectoryReader.open(directory))
        self.parser[lang]   = QueryParser("context", analyzer)
        self.searcher[lang].setSimilarity(self.similarity)
        self.translator.add_language(lang)

    def run(self, lang=None):
        while True:
            command = raw_input("Query:")
            if command == '':
                return
            print("Searching for:", command)
            scoreDocs = self.query[lang](command, 3)
            self.printResult(scoreDocs)

    def printResult(self, scoreDocs, lang):
        print("Number of retrieved documents:", len(scoreDocs))
        for scoreDoc in scoreDocs:
            doc = self.searcher[lang].doc(scoreDoc.doc)
            for id in doc.getFields('id'):
                print("Id:", id.stringValue())
            print("Score:", scoreDoc.score)
            print("Name:", doc.get("docname").encode('utf-8'))
            print("Context:", doc.get("context").encode('utf-8'))

    def getDoc(self, scoreDoc, lang):
        return self.searcher[lang].doc(scoreDoc.doc)

    def queryTest(self, command, lang):
        self.printResult(self.query(command, lang, 5))

    def query(self, command, lang, n=50):
        esccommand = self.parser[lang].escape(command)
        query = self.parser[lang].parse(esccommand)
        scoreDocs = self.searcher[lang].search(query, n).scoreDocs
        return scoreDocs

    def queryMulti(self, command, lang, n=50, p=1):
        """ Returns scored documents in multiple languages.

        Parameters:
        command (str): query string
        lang    (str): language in which is the query
        n       (int): number of documents retrieved
        p       (float): reduces number of retrieved documents from each language
                         e.g.: for 3 languages, n = 50 and p = 0.5 from each language
                         25 documents will be retrieved.
                         Must satisfy n*len(langs)*p >= n

        Returns:

        [scoreDocs]: ordered list of scored documents by their score

        """
        scoreDocs = []
        for to in self.languages:
            transl_comm = self.translator(lang, to, command)
            scoreDocs.append(self.query(transl_comm, to, int(n*p)))
        return scoreDocs.sort(key=lambda x: x.score, reverse=True)[:n]
