from retrieval_bm25.src import utils
from retrieval_bm25.src import retrieval
from retrieval_bm25.src.argparse import parse_args
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

all_langs = ['en', 'de', 'es']

if __name__ == '__main__':
    args = parse_args()
    # start java VM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    root = utils.get_root()

    datadir = os.path.join(root,'data')
    data = utils.load_data(datadir)
    idxdir  = os.path.join(datadir, 'indexes')
    if args.index == None:
        idxdir  = os.path.join(datadir, args.index)
    elif arg.language != 'all':
        idxdir  = os.path.join(datadir, args.language)

    if args.create:
        indexer.createIndexes(data, args.dataset, args.language)
    if args.query != None:
        #if not os.path.isfile(idxfile):
        #    raise Exception("Could not find indexfile: {}".format(idxfile))
        if args.analyzer == None or args.language == 'all':
            raise ValueError("To retrieve query you must specify analyzer and language")
        analyzer = analyzers['args.analyzer']()
        searcher = retrieval.Searcher(idxdir, analyzer)
        searcher.queryTest(args.query)
    if args.hitk:
        if args.language == 'all':
            for lang in all_langs:
                analyzer = analyzers[lang]()
                searcher = retrieval.Searcher(idxdir, analyzer)
                searcher.hitAtK(data=data, dataset=args.dataset,
                    langContext=args.language, langQuestion=args.language, k=100)
        else:
            if args.analyzer == None:
                analyzer = analyzers[args.language]()
            else:
                analyzer = analyzers[args.analyzer]()
            searcher = retrieval.Searcher(idxdir, analyzer)
            searcher.hitAtK(data=data, dataset=args.dataset,
                    langContext=args.language, langQuestion=args.language, k=100)
    if args.hit_dist:
        raise NotImplementedError("This has not been implemented yet")

def createIndexes(data, idxdir, args):
    if args.language == 'all':
        for lang in all_langs:
            idxdir_curr = os.path.join(idxdir, lang)
            createIndex(data, idxdir_curr, lang, args)

def createIndex(data, idxdir, lang, args):
    if args.analyzer == None:
        analyzer = analyzers[lang]()
    else:
        analyzer = analyzers[args.analyzer]()
    indexer = retrieval.Indexer(idxdir, analyzer)
    indexer.createIndex(data, args.dataset, lang)

