import os
import re
import string
from glob import glob
import json
import requests
from zipfile import ZipFile
import sys
import codecs

url_MLQA_V1 = 'https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip'

try:
    import git
except:
    pass


def get_root():
    try:
        root = git.Repo(os.getcwd(), search_parent_directories=True).git.rev_parse('--show-toplevel')
    except:
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return root

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    return

def download_data(datadir):
    req = requests.get(url_MLQA_V1, allow_redirects=True)
    zf = os.path.join(datadir,"MLQA_V1.zip")

    if not os.path.isfile(zf):
        print('Downloading')
        download_url(url_MLQA_V1, zf)
    if not os.path.isdir(os.path.join(datadir, "MLQA_V1")):
        print('Extracting...')
        with ZipFile(zf, 'r') as zipObj:
            zipObj.extractall(datadir)
    return

def datasets2files(datadir,data):
    datadir_dev  = os.path.join(datadir,"mlqa_dev")
    datadir_test = os.path.join(datadir,"mlqa_test")
    print(datadir_dev)
    print(datadir_test)
    if not os.path.exists(datadir_dev):
        os.makedirs(datadir_dev)
    if not os.path.exists(datadir_test):
        os.makedirs(datadir_test)

    dataset2files(datadir_dev,data['mlqa_dev'])
    dataset2files(datadir_test,data['mlqa_test'])

def dataset2files(datadir, dataset):
    for lang, data in dataset.items():
        if lang[-2:] != lang[-14:-12]:
            continue
        langdir = os.path.join(datadir, lang)
        print('Processing ', lang)
        counter = 0
        if not os.path.exists(langdir):
            os.makedirs(langdir)
        for doc in data['data']:
            title = doc['title']
            for paragraph in doc['paragraphs']:
                counter += 1
                fname = os.path.join(langdir,"p{:04}.txt".format(counter))
                if os.path.exists(fname):
                    continue
                with codecs.open(fname,'w+', 'utf-8') as fp:
                    fp.write(title)
                    fp.write("\n")
                    fp.write(paragraph['context'])
                    #fp.write(title.encode("utf-8"))
                    #fp.write("\n".encode("utf-8"))
                    #fp.write(paragraph['context'].encode("utf-8"))if __name__ == '__main__':

    def load_data(datadir, squad=False):
        data = {
        'mlqa_test':{},
        'mlqa_dev':{},
        'transl_test':{},
        'transl_train':{},
        }

        for file in glob(datadir+'/MLQA_V1/test/*.json'):
            with open(file) as fp:
                data['test'][os.path.basename(file)[:-5]] = json.load(fp)

        for file in glob(datadir+'/MLQA_V1/dev/*.json'):
            with open(file) as fp:
                data['dev'][os.path.basename(file)[:-5]] = json.load(fp)

        # squad datasets
        if squad:
            for file in glob(datadir+'/mlqa-translate-test/*.json'):
                with open(file) as fp:
                    data['transl_test'][os.path.basename(file)[:-5]] = json.load(fp)

            for file in glob(datadir+'/mlqa-translate-train/*.json'):
                with open(file) as fp:
                    data['transl_train'][os.path.basename(file)[:-5]] = json.load(fp)
        return data


def normalize_text(s):
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
