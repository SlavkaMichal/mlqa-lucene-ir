from src.reader import Reader
#from src import metrics
from src.argparse import parse_args
from src.datasets import MLQADataset
from src.translator import Translator
import string
from collections import Counter
import cProfile
import regex as re
import json
import argparse
import config
import pickle
import socket
import os
import sys
import numpy as np

all_langs = ['en', 'de', 'es']

def main(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for k, v in vars(args).items():
        print("{0: <12}: {1}".format(k,v))
    print("{0: <12}: {1}".format("Hostname",socket.gethostname()))

    if args.dry_run:
        sys.exit(0)

    sock.connect((args.server, args.port))
    #pr = cProfile.Profile()
    #pr.enable()
    f1 = runf1(sock, args)
    #ipr.disable()
    #pr.print_stats()

    sock.close()

    return f1

def recvall(conn):
    data = conn.recv(4096)
    n = pickle.loads(data)
    data = data[len(pickle.dumps(n)):]

    while n != len(data):
        data += conn.recv(4096)

    return pickle.loads(data)

def sendall(conn, msg):
    data = pickle.dumps(msg)
    n = len(data)
    data = pickle.dumps(n) + data
    conn.sendall(data)
    return

def init(conn, dataset, args):
    print("Initializing")
    msg = {'init':{
        'dataset':dataset,
        'n':args.topk,
        'langs':[args.langSearch],
        'b':args.b,
        'k1':args.k1
        }}
    for k, v in msg['init'].items():
        print("{0: <12}: {1}".format(k,v))
    sendall(conn, msg)

def close(conn, stop=False):
    print("Stoping")
    msg = {'stop':stop}
    sendall(conn, msg)
    conn.close()

def search(conn, question, lang):
    msg = {'search':[{'lang':lang, 'question':question, 'id':0}]}
    sendall(conn, msg)

def runf1(conn, args):
    # evaluation dataset
    # english context so that answer is in english
    data = MLQADataset(args.dataset, 'en', args.langQuestion)

    # initialize searcher
    init(conn, 'wiki', args)

    # initialise reader
    print("Reader")
    reader = Reader(model="models/distilbert-base-uncased-distilled-squad/",
            tokenizer="models/distilbert-uncased-my-tok")

    # initialise translator
    print("Translator")
    languages = {args.langQuestion, args.langSearch, 'en'}
    translator = Translator(languages)
    print("Translating between: {}".format(str(languages)))
    counters = {
            'f1':[],
            'tally':0,
            'score':[]}


    for doc in data.get():
        questionSearch = translator(doc['question'], args.langQuestion, args.langSearch)
        #print("questionSearch ", questionSearch.encode('utf-8'))
        search(conn, questionSearch, args.langSearch)

        if args.langSearch == 'en':
            questionRead = questionSearch
        else:
            questionRead = translator(doc['question'], args.langQuestion, 'en')
        #print("questionRead ", questionRead.encode('utf-8'))
        # recv = {'search':[{'id':qid, 'docs':[{'context':'...', 'title':'...', 'score':score}]}]
        bestScore = 0
        recv = recvall(conn)
        for n, docSearch in enumerate(recv['search'][0]['docs']):
            # reader answer question given contexts
            #print("n: ", n)
            #print("contextSearch ", docSearch['context'].encode('utf-8'))
            contextRead = translator(docSearch['context'], args.langSearch, 'en')
            #print("contextRead ", contextRead.encode('utf-8'))
            _, answerRead, score = reader(questionRead, contextRead)
            if score >= bestScore:
                bestScore   = score
                bestAnswer  = answerRead
                bestContext = contextRead

        #print("goldAnswer: ",doc['answer'].encode('utf-8'))
        #print("Answer:     ",bestAnswer.encode('utf-8'))
        counters['f1'].append(f1_drqa(bestAnswer, doc['answer']))
        counters['tally'] += 1
        counters['score'].append(bestScore)
        # test
        if args.stop != 0 and counters['tally'] >= args.stop:
            print("Stoping at: ",counters['tally'])
            break
        #if i > 1:
        #    break

    f1 = np.array(counters['f1'])
    exact_match = f1[f1 == 1.0].sum()/f1.size
    print("Exact match: {}".format(exact_match))
    print("F1 mean: {}".format(f1.mean()))
    print("Mean score: {}".format(sum(counters['score'])/counters['tally']))
    print("Total: {}".format(counters['tally']))
    if args.save_as:
        print("Writing to: ", args.save_as)
        with open(args.save_as, "w") as fp:
            json.dump(counters, fp)

    close(conn, args.stop_server)

    return f1.mean()

# f1 score from DrQA
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_drqa(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Searching server')
    parser.add_argument('-p', '--port', action='store', type=int, default=config.port,
                        help='TCP port')
    parser.add_argument('--stop', action='store', type=int, default=config.stop,
                        help='Stop early')
    parser.add_argument('-s', '--server', action='store', type=str, default=config.server,
                        help='TCP port')
    parser.add_argument('-d', '--dataset', action='store', type=str, default=config.dataset,
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
    parser.add_argument('--stop-server', action='store_true',
                        help='Stop server after evaluation')
    parser.add_argument('-b', action='store', type=int, default=config.b,
                        help='Stop early')
    parser.add_argument('--k1', action='store', type=int, default=config.k1,
                        help='Stop early')
    main(parser.parse_args())
