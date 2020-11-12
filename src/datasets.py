from .utils import get_root, get_dataname

class MLQA_Dataset():
    def __init__(self, dataset, langContext, langQuestion):
        root = get_root()
        datadir = os.path.join(root, 'data')
        name = get_dataname(dataset, langContext, langQuestion)
        self.data = load_data(datadir)
        self.data = self.data['mlqa_'+dataset][name]['data']

    def get(self):
        for doc in self.data:
            for paragraph in doc['paragraphs']:
                for qa in paragraph['qas']:
                    yield {
                        'title':doc['title'],
                        'context':paragraph['context'],
                        'question':qa['question'],
                        'qid':qa['id'],
                        'answer':qa['answers'][0]['text'],
                        'start':qa['answers'][0]['answer_start'],
                        }

class EsWiki():
    def __init__(self, dppath):
        root = get_root()
        datadir = os.path.join(root, 'data')
        name = get_dataname(dataset, langContext, langQuestion)
        self.data = load_data(datadir)
        self.data = self.data['mlqa_'+dataset][name]['data']

    def get(self):
        yield {
            'title':doc['title'],
            'context':paragraph['context'],
            'question':qa['question'],
            'qid':qa['id'],
            'answer':qa['answers'][0]['text'],
            'start':qa['answers'][0]['answer_start'],
            }
