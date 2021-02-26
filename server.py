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
from time import time


all_langs = ['en', 'de', 'es']

def main(args):
    for k, v in vars(args).items():
        print("{0: <12}: {1}".format(k,v))
    print("{0: <12}: {1}".format("Hostname",socket.gethostname()))
    if args.dry_run:
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
        if run(conn, addr, args):
            sock.close()
            break

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

def run(conn, addr, args):
    #stop_search = time()
    cnt = 0
    while (True):
        start_recv = time()
        recv = recvall(conn)
        stop_recv = time()

        if 'init' in recv:
            print("Initialising searcher")
            b = recv['init']['b']
            k1 = recv['init']['k1']
            searcher = Searcher(k1=k1, b=b)
            print("b: ", b)
            print("k1: ", k1)
            print("index dir: ", args.index_dir)

            n = recv['init']['n']
            dataset =  recv['init']['dataset']
            print("dataset: ", dataset)
            print("top k: ", n)
            print('Search languages', recv['init']['langs'])
            for lang in recv['init']['langs']:
                searcher.addLang(lang, dataset, lang, args.index_dir)

        if 'search' in recv:
            res = {}
            res['search'] = []
            for s in recv['search']:
                documents = {'id': s['id'],'docs':[]}
                scoreDocs = searcher.query(s['question'], s['lang'], n)
                for scoreDoc in scoreDocs:
                    document = {}
                    doc = searcher.getDoc(scoreDoc)
                    document['context'] = doc.get("context")
                    document['title'] = doc.get("title")
                    document['score']   = scoreDoc.score
                    documents['docs'].append(document)
                res['search'].append(documents)
                sendall(conn, res)
            #print("Searching time: ", stop_search-start_search)

        if 'stop' in recv:
            print("Stoping")
            if recv['stop']:
                conn.close()
                return 1
            else:
                conn.close()
            break

        end_request = time()
        if cnt % args.write_intensity == 0 and args.write_intensity != 0:
            print("Request number: ", cnt)
            print("Recv took:   ", stop_recv-start_recv)
            print("Reqest took: ", end_request-stop_recv)
            print("Reqest type: ", recv.keys())
        cnt +=1
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Searching server')
    parser.add_argument('-p', '--port', action='store', type=int, default=config.port,
                        help='TCP port')
    parser.add_argument('-w', '--write-intensity', action='store', type=int, default=0,
                        help='Search stats frequency')
    parser.add_argument('-i', '--index-dir', action='store', type=str, default=None,
                        help='Index directory')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='Print configuration')
    main(parser.parse_args())


