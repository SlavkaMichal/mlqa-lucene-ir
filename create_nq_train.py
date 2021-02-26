from src.connection import Connection
import jsonlines
from src.utils import normalize_text
from src.config import Config
import argparse
import re
from src.datasets import NQSimplified


def main(cfg):
    if cfg.test:
        data_train = NQSimplified('test')
        data_valid = NQSimplified('test')
    else:
        data_train = NQSimplified('train')
        data_valid = NQSimplified('valid')

    conn = Connection()

    # nq is in english so search english wikipedia
    conn.init_client(cfg.server, 'wiki', cfg.topk, ['en'], cfg.search_params)

    create_nq_train(conn, data_train.get(), cfg)
    create_nq_train(conn, data_valid.get(), cfg)
    conn.close(True)


def create_nq_train(conn, data, config):
    """
    nq_q -> nq_train_q, nq_train_paragraphs
    """
    # number of words around searched query
    context_len = 7
    writer = jsonlines.open(config.save_as, mode='w')
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    for d in data:
        print("-"*40)
        train_sample = dict(
            nq=d,
            question=d['question_text'],
            docs_negative=[],
            docs_pn=[],
            docs_positive=[],
            nq_id=d['example_id'])

        # remove duplicate spans
        answers = list(dict.fromkeys([tuple(a) for a in d['answers']]))

        for (start, end) in answers:
            print(start, end)
            answer_text = " ".join(d['document_text'].split(' ')[start:end])

            print("Answer: ", answer_text)
            # search for documents containing answer
            span_start = start - context_len if start - context_len > 0 else 0
            span_end = end + context_len
            search_paragraph = " ".join(d['document_text'].split(' ')[span_start: span_end])
            search_paragraph = normalize_text(re.sub(cleanr, '', search_paragraph))
            print("Searching for: ", search_paragraph)
            print("Span:", span_start, span_end)
            print("Answer: ", answer_text)
            print("Span:", start, end)
            conn.search(search_paragraph+d['question_text'], 'en', 'context')
            documents = conn.recvall()['search'][0]['docs']

            print("N documents:", len(documents))
            for doc in documents:
                #print("--------Document--------")
                title_norm = normalize_text(doc['title'])
                #print("Retrieved title: ", title_norm)
                title_len = len(title_norm)
                title_nq = normalize_text(d['document_text'][:3*title_len])
                #print("Nq title: ", title_nq)
                context_norm = normalize_text(doc['context'])
                #print("Retrieved context:", context_norm)
                #print("Tile in:", title_norm in title_nq)
                #print("Search para in:", search_paragraph in context_norm)
                # check if retrieved document has the same title and contains answer
                if title_norm in title_nq and search_paragraph in context_norm:
                    print("CORRECT")
                    answer_start = doc['context'].find(answer_text)
                    train_sample['hit'] = dict(title=doc['title'], context=doc['context'], score=doc['score'],
                                               answer={'text': answer_text, 'start': answer_start})
                    break
                elif answer_text in doc['context']:
                    print("CORRECT2")
                    answer_start = doc['context'].find(answer_text)
                    train_sample['docs_positive'].append(
                    {'title': doc['title'],
                     'context': doc['context'],
                     'score': doc['score'],
                     'answer': {'text': answer_text, 'start': answer_start}})
                else:
                #print("INcorrect")
                    train_sample['docs_pn'].append(
                        {'title': doc['title'],
                         'context': doc['context'],
                         'score': doc['score'],
                         })

            # search question for documents using question as query
            conn.search(d['question_text'], 'en', 'context')
            documents_negative = conn.recvall()['search'][0]['docs']
            # find negative documents
            for doc in documents_negative:
                train_sample['docs_negative'].append(
                    {'context': doc['context'],
                     'title': doc['title'],
                     'score': doc['score']
                     })
        writer.write(train_sample)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Searching server')
    parser.add_argument('-c', '--config',  action='store', type=str, default="configs/create_nq_train.json",
                        help='Configuration file')
    parser.add_argument('--dump-config',   action='store', type=str, help="")
    parser.add_argument('-p', '--port',    action='store', type=int, help='TCP port')
    parser.add_argument('-s', '--server',  action='store', type=str, help='Hostname or address')
    parser.add_argument('-d', '--dataset', action='store', type=str, help='Evaluation dataset')
    parser.add_argument('-f', '--save-as', action='store', type=str, default=None, help='Save result')
    parser.add_argument('-k', '--topk',    action='store', type=int, help='Retrieve k contexts')
    parser.add_argument('--test',          action='store', type=int, help='Number of test steps')
    parser.add_argument('-n', '--dry-run', action='store_true', help='Print configuration and exit')
    parser.add_argument('--b',             action='store', type=int, help='Search parameter')
    parser.add_argument('--k1',            action='store', type=int,  help='Search parameter')

    args = parser.parse_args()
    config = Config(args)
    print(config)

    main(config)
