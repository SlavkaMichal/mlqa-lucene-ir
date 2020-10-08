import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Creating and Searching index files')

    parser.add_argument('-i', '--index', action='store', type=str, default=None,
                        required=True, help='Path to index file')
    parser.add_argument('-d', '--dataset', action='store', type=str, default='dev',
                        choices=['dev', 'test'], help='Dataset for indexing')
    parser.add_argument('-l', '--language', action='store', type=str, default='en',
                        choices=['en', 'es', 'de','multi'], help='Context language')
    parser.add_argument('-a', '--analyzer', action='store', type=str, default='standard',
                        choices=['en', 'es', 'de','standard'], help='Select analyzer')
    parser.add_argument('-q', '--query', action='store', type=str, default=None,
                        help='Query data')
    parser.add_argument('-c', '--create', action='store_true',
                        help='Create new index')
    parser.add_argument('-e', '--eval', action='store_true',
                        help='Evaluate hitrate')

    parser.add_argument('-p', '--progress_bar', action='store_true',
                        help='Show progress bar while indexing TODO')
    parser.add_argument('--test', action='store_true',
                        help='Test run TODO')

    return parser.parse_args()

