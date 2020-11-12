from src import utils
from src.reader import Reader
from src.retrieval import Indexer, Searcher
from src import metrics
from src.argparse import parse_args
import lucene
import os

all_langs = ['en', 'de', 'es']

if __name__ == '__main__':
    args = parse_args()
    # start java VM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    if args.create:
        langs = [args.language]
        if args.language == 'all':
            langs = all_langs
        for lang in langs:
            if args.analyzer == None:
                analyzer = lang
            indexer = Indexer(lang, args.dataset, analyzer)
            indexer.createIndex()

    if args.query != None:
        #if not os.path.isfile(idxfile):
        #    raise Exception("Could not find indexfile: {}".format(idxfile))
        if args.analyzer == None or args.language == 'all':
            raise ValueError("To retrieve query you must specify analyzer and language")
        searcher = Searcher(
                lang=args.language,
                analyzer=args.analyzer,
                dataset=args.dataset)
        searcher.queryTest(args.query)

    if args.run == 'reader':
        reader = Reader()
        reader.run(
                lang=args.lang,
                analyzer=args.analyzer,
                dataset=args.dataset)

    if args.metric == 'dist':
        metrics.hits(
               dataset=args.dataset,
               langContext=args.language,
               langQuestion=args.language,
               distant=True,
               k=50)

    if args.metric == 'hit@k':
        metrics.hits(
               dataset=args.dataset,
               langContext=args.language,
               langQuestion=args.language,
               distant=False,
               k=50)

    if args.metric == 'qa_f1':
        metrics.qa_f1(
               dataset=args.dataset,
               langContext=args.language,
               langQuestion=args.language,
               k=10)

    if args.metric == 'review':
        metrics.review(
               dataset=args.dataset,
               langContext=args.language,
               langQuestion=args.language,
               k=10)
