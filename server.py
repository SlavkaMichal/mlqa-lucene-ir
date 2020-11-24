from src import utils
from src.reader import Reader
from src.retrieval import Indexer, Searcher
from src import metrics
from src.argparse import parse_args
import argparse
import config
import pickle
import socket
import lucene
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
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])


    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # bind socket to a port
    sock.bind((socket.gethostname(), args.port))
    print("Host name: {}".format(socket.gethostname()))
    print("Port: {}".format(args.port))
    sock.listen()

    while (True):
        (conn, addr) = sock.accept()
        print("Connected to {}", addr)
        if run(conn, addr):
            sock.close()
            break

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

def run(conn, addr):
    while (True):
        recv = recvall(conn)
        res = {}
        n = 1

        if 'init' in recv:
            print("Initialising searcher")
            searcher = Searcher()
            dataset =  recv['init']['dataset']
            n = recv['init']['n']
            for lang in recv['init']['langs']:
                searcher.addLang(lang, dataset, lang)
            res['init'] = True

        if 'search' in recv:
            res['search'] = []
            for s in recv['search']:
                documents = {'id': s['id'],'docs':[]}
                scoreDocs = searcher.query(s['question'], s['lang'], n)
                for scoreDoc in scoreDocs:
                    document = {}
                    doc = searcher.getDoc(scoreDoc)
                    document['context'] = doc.get("context")
                    document['title'] = doc.get("docname")
                    document['score']   = scoreDoc.score
                    contexts['docs'].append(document)
                res['search'].append(documents)

        if 'stop' in recv:
            if recv['stop']:
                if res != {}:
                    sendall(conn, res)
                conn.close()
                break

        if 'stopall' in recv:
            if recv['stopall']:
                if res != {}:
                    sendall(conn, res)
                conn.close()
                return 1
        sendall(conn, res)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Searching server')
    parser.add_argument('-p', '--port', action='store', type=int, default=config.port,
                        help='TCP port')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='Print configuration')
    main(parser.parse_args())


