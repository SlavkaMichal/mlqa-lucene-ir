import sys, os, lucene, threading, time
from datetime import datetime
from src.utils import get_root, load_data
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
    def dataname(self, dataset, context, question):
        return dataset+'-context-'+context+'-question-'+question


class Indexer(Retriever):
    def __init__(self, storedir, analyzer, datadir=None):
        if not os.path.exists(storedir):
            os.mkdir(storedir)
        # stores index files, poor concurency try NIOFSDirectory instead
        store = SimpleFSDirectory(Paths.get(storedir))
        # limit max. number of tokens while analyzing
        self.analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        # configuration for index writer
        config = IndexWriterConfig(analyzer)
        # creates or overwrites index
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        # setting similarity BM25Similarity(k1=1.2,b=0.75)
        similarity = BM25Similarity()
        config.setSimilarity(similarity)
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

        if datadir != None:
            print("Indexing files")
            indexDocs(datadir, writer)
            writer.commit()
            writer.close()
            print("done")

    def addDoc(self, ids, title, context):
        print("adding:", title.encode('utf-8'))
        doc = Document()
        for i in ids:
            doc.add(Field("id", i, self.ftmeta))
        doc.add(Field("docname", title, self.ftdata))
        doc.add(Field("context", context, self.ftdata))
        self.writer.addDocument(doc)

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
    def __init__(self, index, analyzer):
        directory = SimpleFSDirectory(Paths.get(index))
        similarity = BM25Similarity()
        self.searcher = IndexSearcher(DirectoryReader.open(directory))
        self.searcher.setSimilarity(similarity)
        self.analyzer = analyzer
        self.parser   = QueryParser("context", self.analyzer)

    def run(self):
        while True:
            command = raw_input("Query:")
            if command == '':
                return
            print("Searching for:", command)
            scoreDocs = self.query(command, 3)
            self.printResult(scoreDocs)

    def printResult(self, scoreDocs):
        print("Number of retrieved documents:", len(scoreDocs))
        for scoreDoc in scoreDocs:
            doc = self.searcher.doc(scoreDoc.doc)
            for id in doc.getFields('id'):
                print("Id:", id.stringValue())
            print("Score:", scoreDoc.score)
            print("Name:", doc.get("docname").encode('utf-8'))
            print("Context:", doc.get("context").encode('utf-8'))

    def queryTest(self, command):
        #pdb.set_trace()
        self.printResult(self.query(command, 5))

    def query(self, command, n):
        esccommand = self.parser.escape(command)
        query = self.parser.parse(esccommand)
        scoreDocs = self.searcher.search(query, n).scoreDocs
        return scoreDocs

    def hitAtK(self, data, dataset, langContext, langQuestion, saveas=None, k=50):
        dataname = self.dataname(dataset, langContext, langQuestion)
        data = data['mlqa_'+dataset][dataname]['data']

        root = get_root()
        if saveas == None:
            saveas = os.path.join(root,"data/stats/{}-C{}-Q{}"
                    .format(dataset, langContext, langQuestion))
        else:
            saveas = os.path.join(root,"data/stats/{}".format(saveas))
        print("Saving stats as {}".format(saveas))
        # counters
        dtype = np.dtype([('total', 'i4'), ('hits', 'i8',(k)), ('scores', 'f8', (k))])
        misses = []
        tally = np.zeros(1, dtype=dtype)[0]
        for doc in data:
            #title = doc['title']
            for paragraph in doc['paragraphs']:
                for qa in paragraph['qas']:
                    tally['total'] += 1
                    # TODO it looks like the list is ordered by score
                    # but should not be trusted
                    scoreDocs = self.query(qa['question'], k)
                    prevScore = math.inf
                    qid = qa['id']
                    hit = False
                    for n, scoreDoc in enumerate(scoreDocs):
                        doc = self.searcher.doc(scoreDoc.doc)
                        docIds = [ id.stringValue() for id in doc.getFields('id')]
                        # sanity check
                        # TODO remove condition if it works
                        if prevScore < scoreDoc.score:
                            print("Previous score: {} was smaller than current score: {}".
                                    format(prevScore, scoreDoc.score))
                            self.printResult(scoreDocs)
                            return
                        prevScore = scoreDoc.score
                        # check if the document is a hit
                        if qid in docIds:
                            tally['hits'][n] += 1
                            tally['scores'][n] += scoreDoc.score
                            hit = True
                            break
                    if not hit:
                        misses.append({'question':qa['question'],
                            'context' : paragraph['context']})

        with open(saveas+"-misses.json", "w+") as fp:
            json.dump(misses, fp)
        np.save(saveas+".npy", tally)
        print("Evaluation of retrieval done")
        return

if __name__ == '__main__':
    # init java VM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    start = datetime.now()
    root = get_root()
    print('root',root)
    try:
        datadir = os.path.join(root, 'data/mlqa_dev/dev-context-en-question-en/')
        storedir = os.path.join(root, 'data/mlqa_dev/c-en-q-en.index')
        indexer(storedir, StandardAnalyzer(),datadir)
        end = datetime.now()
        print("Indexing took: ", end-start)
    except Exception as e:
        print("Failed: ", e)
        raise e
