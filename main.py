from src import utils
from src import indexer as idx
from src.argparse import parse_args
import lucene
from org.apache.lucene.analysis.standard import StandardAnalyzer
import os

analyzers = {
        'en':StandardAnalyzer
        }

if __name__ == '__main__':
    args = parse_args()
    # start java VM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    root = utils.get_root()

    datadir = os.path.join(root,'data')
    idxfile = os.path.join(datadir, args.index)
    analyzer = analyzers[args.language]()
    if args.create:
        indexer = idx.indexer(idxfile, analyzer)
        indexer.createIndex(datadir, args.dataset, args.language)
    if args.query != None:
        #if not os.path.isfile(idxfile):
        #    raise Exception("Could not find indexfile: {}".format(idxfile))
        searcher = idx.searcher(idxfile, analyzer)
        searcher.query(args.query)



