import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Creating and Searching index files')

    #parser.add_argument('-i', '--index', action='store', type=str, default=None,
    #                    help='Path to index file')
    parser.add_argument('-d', '--dataset', action='store', type=str, required=True,
                        choices=['dev', 'test'], help='Dataset for indexing')
    parser.add_argument('-l', '--language', action='store', type=str, required=True,
                        choices=['en', 'es', 'de','multi'], help='Context language')
    parser.add_argument('-a', '--analyzer', action='store', type=str, default=None,
                        choices=['en', 'es', 'de','standard'], help='Select analyzer')
    parser.add_argument('-q', '--query', action='store', type=str, default=None,
                        help='Query data')
    parser.add_argument('-c', '--create', action='store_true',
                        help='Create new index')
    parser.add_argument('-m', '--metric', action='store', type=str,
                        choices=['dist', 'hit@k', 'qa_f1', 'review'], help='Compute metric')
    parser.add_argument('-r', '--run', action='store', type=str,
                        choices=['reader', 'retriever'], help='Interactive')

    parser.add_argument('-p', '--progress_bar', action='store_true',
                        help='Show progress bar while indexing TODO')
    parser.add_argument('--test', action='store_true',
                        help='Test run TODO')

    return parser.parse_args()

