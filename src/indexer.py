import sys, os, lucene, threading, time
from datetime import datetime
from src.utils import get_root, load_data
from glob import glob
import pdb
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

class indexer(object):
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
        self.ftmeta.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

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

    def createIndex(self, datadir, dataset, lang):
        language = dataset+'-context-'+lang+'-question-'+lang
        data = load_data(datadir)['mlqa_'+dataset][language]['data']
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

class searcher(object):
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
            query = self.parser.parse(command)
            scoreDocs = self.searcher.search(query, 10).scoreDocs
            self.printResult(scoreDocs)

    def printResult(self, scoreDocs):
        for scoreDoc in scoreDocs:
            doc = self.searcher.doc(scoreDoc.doc)
            print("Id:", doc.get("id"))
            print("Name:", doc.get("docname"))
            print("Context:", doc.get("context"))

    def query(self, command):
        query = self.parser.parse(command)
        scoreDocs = self.searcher.search(query, 10).scoreDocs
        #self.printResult(scoreDocs)
        print("Number of matched documets:", len(scoreDocs))
        #pdb.set_trace()
        for scoreDoc in scoreDocs:
            doc = self.searcher.doc(scoreDoc.doc)
            print("Id:", doc.get("id").encode('utf-8'))
            print("Name:", doc.get("docname").encode('utf-8'))
            print("Context:", doc.get("context").encode('utf-8'))

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
