from src import utils
from src import retrieval
from src.argparse import parse_args
import lucene
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.de import GermanAnalyzer
from org.apache.lucene.analysis.es import SpanishAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
import os

analyzers = {
        'standard':StandardAnalyzer,
        'en':EnglishAnalyzer,
        'es':SpanishAnalyzer,
        'de':GermanAnalyzer
        }

if __name__ == '__main__':
    args = parse_args()
    # start java VM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    root = utils.get_root()

    datadir = os.path.join(root,'data')
    idxfile = os.path.join(datadir, args.index)
    analyzer = analyzers[args.analyzer]()
    data    = utils.load_data(datadir)
    if args.create:
        indexer = retrieval.Indexer(idxfile, analyzer)
        indexer.createIndex(data, args.dataset, args.language)
    if args.query != None:
        #if not os.path.isfile(idxfile):
        #    raise Exception("Could not find indexfile: {}".format(idxfile))
        searcher = retrieval.Searcher(idxfile, analyzer)
        searcher.queryTest(args.query)
    if args.eval:
        searcher = retrieval.Searcher(idxfile, analyzer)
        searcher.hitAtK(data=data, dataset=args.dataset,
                langContext=args.language, langQuestion=args.language, k=100)




