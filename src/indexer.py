import sys, os, lucene, threading, time
from datetime import datetime
from utils import get_root
from glob import glob

from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search.similarities import BM25Similarity


def indexFiles(datadir, storedir, analyzer):
    if not os.path.exists(storedir):
        os.mkdir(storedir)

    # stores index files, poor concurency try NIOFSDirectory instead
    store = SimpleFSDirectory(Paths.get(storedir))
    # limit max. number of tokens while analyzing
    analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
    # configuration for index writer
    config = IndexWriterConfig(analyzer)
    # creates or overwrites index
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    # setting similarity BM25Similarity(k1=1.2,b=0.75)
    similarity = BM25Similarity()
    config.setSimilarity(similarity)
    # create index writer
    writer = IndexWriter(store, config)

    print("Indexing files")
    indexDocs(datadir, writer)
    writer.commit()
    writer.close()
    print("done")

def indexDocs(datadir, writer):
    ft = FieldType()
    # IndexSearcher will return value of the field
    ft.setStored(True)
    # will be analyzed by Analyzer
    ft.setTokenized(True)
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
    ft.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

    #for datadir, dirnames, filenames in os.walk(root):
    for filename in glob(datadir):
        if not filename.endswith('.txt'):
            continue
        print("adding:", filename)
        break
        path = os.path.join(datadir, filename)
        print("adding:", path)
        with open(path, 'r', encoding='utf-8') as fp:
            contents = fp.read()
        contents = contents.splitlines()
        doc = Document()
        print("title", contents[0].encode('utf-8'))
        doc.add(Field("docname", contents[0], ft))
        doc.add(Field("context", contents[1], ft))
        writer.addDocument(doc)

if __name__ == '__main__':
    # init java VM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    start = datetime.now()
    root = get_root()
    print('root',root)
    try:
        datadir = os.path.join(root, 'data/mlqa_dev/dev-context-en-question-en/')
        storedir = os.path.join(root, 'data/mlqa_dev/c-en-q-en.index')
        indexFiles(datadir, storedir, StandardAnalyzer())
        end = datetime.now()
        print("Indexing took: ", end-start)
    except Exception as e:
        print("Failed: ", e)
        raise e
