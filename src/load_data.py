from glob import glob
import os
import json

def load_data(datadir,squad=False):
    # squad datasets
    data = {
    'mlqa_test':{},
    'mlqa_dev':{},
    'transl_test':{},
    'transl_train':{},
    }

    for file in glob(datadir+'/MLQA_V1/test/*.json'):
        with open(file) as fp:
            data['mlqa_test'][os.path.basename(file)[:-5]] = json.load(fp)

    for file in glob(datadir+'/MLQA_V1/dev/*.json'):
        with open(file) as fp:
            data['mlqa_dev'][os.path.basename(file)[:-5]] = json.load(fp)

    # squad datasets
    if squad:
        for file in glob(datadir+'/mlqa-translate-test/*.json'):
            with open(file) as fp:
                data['transl_test'][os.path.basename(file)[:-5]] = json.load(fp)

        for file in glob(datadir+'/mlqa-translate-train/*.json'):
            with open(file) as fp:
                data['transl_train'][os.path.basename(file)[:-5]] = json.load(fp)
    return data
