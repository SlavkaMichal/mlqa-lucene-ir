from src.reader import Reader
from src.datasets import MLQADataset
from src.translator import Translator
import string
import math
from collections import Counter
import regex as re
import json
import argparse
from src.config import Config
from src.connection import Connection
import socket
import sys
import numpy as np

all_langs = ['en', 'de', 'es']


def main(args):
    for k, v in vars(args).items():
        print("{0: <12}: {1}".format(k, v))
    print("{0: <12}: {1}".format("Hostname", socket.gethostname()))

    if args.dry_run:
        sys.exit(0)

    f1 = run_f1(args)

    return f1


def run_f1(args):
    # evaluation dataset
    # english context so that answer is in english
    data = MLQADataset(args.dataset, 'en', args.langQuestion)

    # initialize searcher
    conn = Connection()
    conn.init_client((args.server, args.port), 'wiki', args.topk, [args.langSearch], dict(b=args.b, k1=args.k1))

    # initialise reader
    print("Reader")
    reader = Reader(model="models/distilbert-base-uncased-distilled-squad/",
                    tokenizer="models/distilbert-uncased-my-tok", select_span='old')

    # initialise translator
    print("Translator")
    languages = {args.langQuestion, args.langSearch, 'en'}
    translator = Translator(languages)
    print("Translating between: {}".format(str(languages)))
    counters = {
        'f1': [],
        'tally': 0,
        'score': []}

    for doc in data.get():
        question_search = translator(doc['question'], args.langQuestion, args.langSearch)
        # print("questionSearch ", questionSearch.encode('utf-8'))
        conn.search(question_search, args.langSearch, field='context')

        if args.langSearch == 'en':
            question_read = question_search
        else:
            question_read = translator(doc['question'], args.langQuestion, 'en')
        # print("questionRead ", questionRead.encode('utf-8'))
        # recv = {'search':[{'id':qid, 'docs':[{'context':'...', 'title':'...', 'score':score}]}]
        best_score = -math.inf
        best_answer = ""
        recv = conn.recvall()
        # if multiple questions were send, another loop would be required
        for n, docSearch in enumerate(recv['search'][0]['docs']):
            # reader answer question given contexts
            # TODO batch translation
            context_read = translator(docSearch['context'], args.langSearch, 'en')
            # print("contextRead ", contextRead.encode('utf-8'))
            _, answer_read, score = reader(question_read, context_read)
            if score >= best_score:
                best_score = score
                best_answer = answer_read

        counters['f1'].append(f1_drqa(best_answer, doc['answer']))
        counters['tally'] += 1
        counters['score'].append(best_score)
        # test
        if args.stop != 0 and counters['tally'] >= args.stop:
            print("Stopping at: ", counters['tally'])
            break
        # if i > 1:
        #    break

    f1 = np.array(counters['f1'])
    exact_match = f1[f1 == 1.0].sum() / f1.size
    print("Exact match: {}".format(exact_match))
    print("F1 mean: {}".format(f1.mean()))
    print("Mean score: {}".format(sum(counters['score']) / counters['tally']))
    print("Total: {}".format(counters['tally']))
    if args.save_as:
        print("Writing to: ", args.save_as)
        with open(args.save_as, "w") as fp:
            json.dump(counters, fp)

    conn.close(args.stop_server)
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
    parser.add_argument('-p', '--port',         action='store', type=int, help='TCP port')
    parser.add_argument('--stop',               action='store', type=int, help='Stop early')
    parser.add_argument('-s', '--server',       action='store', type=str, help='TCP port')
    parser.add_argument('-d', '--dataset',      action='store', type=str, help='TCP port')
    parser.add_argument('-l', '--langSearch',   action='store', type=str, help='Context language')
    parser.add_argument('-q', '--langQuestion', action='store', type=str, help='Question language')
    parser.add_argument('-f', '--save-as',      action='store', type=str, help='Save result', default="")
    parser.add_argument('-k', '--topk',         action='store', type=int, help='Retrieve k contexts')
    parser.add_argument('--b',                  action='store', type=int, help='BM25 hyperparameter')
    parser.add_argument('--k1',                 action='store', type=int, help='BM25 hyperparameter')
    parser.add_argument('-n', '--dry-run',      action='store_true', help='Print configuration')
    parser.add_argument('--stop-server',        action='store_true', help='Stop server after evaluation')
    config = Config(parser.parse_args())
    main(parser.parse_args())
