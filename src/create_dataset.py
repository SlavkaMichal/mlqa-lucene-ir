import requests
import os
from zipfile import ZipFile
from load_data import load_data
import sys
import codecs
try:
    import git
except:
    pass

url_MLQA_V1 = 'https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip'

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

if __name__ == "__main__":
    try:
        root = git.Repo(os.getcwd(), search_parent_directories=True).git.rev_parse('--show-toplevel')
    except:
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    datadir = os.path.join(root, 'data')
    download_data(datadir)
    data = load_data(datadir)
    print("here")
    datasets2files(datadir, data)

