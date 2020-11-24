from src import utils
from src.reader import Reader
from src import metrics
from src.argparse import parse_args
from src.translator import Translator
import argparse
import config
import pickle
import socket
import os
import sys

all_langs = ['en', 'de', 'es']

def main(args):
    if args.dry_run:
        for k, v in vars(args).items():
            print("{0: <12}: {1}".format(k,v))
        print("{0: <12}: {1}".format("Hostname",socket.gethostname()))
        sys.exit(0)
    # start java VM


    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # bind socket to a port
    print("Server name: {}".format(args.server))
    print("Port: {}".format(args.port))

    sock.connect((args.server, args.port))
    runf1(sock, args)
    print("Connected to {}", addr)

    sock.close()

def recvall(conn):
    print("recieving")
    data = conn.recv(4096)
    n = pickle.loads(data)
    print("n",n)
    print("data",data)
    print("type n",type(n))
    data = data[len(pickle.dumps(n)):]
    print("data",data)

    while n != len(data):
        data += conn.recv(4096)

    return pickle.loads(data)

def sendall(conn, msg):
    print("sending")
    data = pickle.dumps(msg)
    n = len(data)
    data = pickle.dumps(n) + data
    sendall(data)
    return

def init(conn, dataset, langs, n):
    msg = {'init':{'dataset':dataset, 'n':n, 'langs':langs}}
    sendall(conn, msg)

def search(conn, questions, lang):
    msg = {'search':[{'lang':lang, 'question':question, 'id':0}]}
    sendall(conn, msg)

def runf1(conn, args):
    # evaluation dataset
    data = MLQADataset(args.dataset, args.langQuestion, 'en')

    # initialize searcher
    init(conn, 'wiki', [args.langSearch], args.topk)

    # initialise reader
    reader = Reader()

    # initialise translator
    languages = {args.langQuestion, args.langSearch, 'en'}
    translator = Translator(languages)
    print("Translating between: {}".format(str(languages)))
    counters = {
            'f1':[],
            'tally':0,
            'score':[]}

    for doc in data.get():
        tally['total'] += 1
        questionSearch = translator(doc['question'], args.langQuestion, args.langSearch)
        recv = search(con, [questionSearch], args.langSearch)
        questionRead   = translator(doc['question'], args.langQuestion, 'en')
        # recv = {'search':[{'id':qid, 'docs':[{'context':'...', 'title':'...', 'score':score}]}]
        bestScore = 0
        for n, docSearch in enumerate(recv['search'][0]['docs']):
            # reader answer question given contexts
            contextRead = translator(docSearch['context'], args.langSearch, 'en')
            _, answerRead, score = reader(questionRead, contextRead)
            if score >= bestScore:
                bestScore   = score
                bestAnswer  = answerRead
                bestContext = contextRead

        counters['f1'].append(utils.f1_drqa(bestAnswer, doc['answer']))
        counters['tally'] += 1
        counters['score'].append(bestScore)

    f1 = np.array(counters['f1'])
    exact_match = f1[f1 == 1.0].sum().f1.size()
    print("Exact match: {}".format(exact_match))
    print("F1 mean: {}".format(f1.mean()))
    print("Mean score: {}".format(sum(counters['score'])/counters['tally']))
    print("Total: {}".format(counters['tally']))
    with open(args.file, "wb") as fp:
        pickle.dump(counters, fp)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Searching server')
    parser.add_argument('-p', '--port', action='store', type=int, default=config.port,
                        help='TCP port')
    parser.add_argument('-s', '--server', action='store', type=str, default=config.server,
                        help='TCP port')
    parser.add_argument('-l', '--langSearch', action='store', type=str, default=config.langSearch,
                        help='Context language')
    parser.add_argument('-q', '--langQuestion', action='store', type=str, default=config.langQuestion,
                        help='Question language')
    parser.add_argument('-f', '--save-as', action='store', type=str, default="",
                        help='Save result')
    parser.add_argument('-k', '--topk', action='store', type=int, default=config.topk,
                        help='Retriev k contexts')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='Print configuration')
    main(parser.parse_args())
