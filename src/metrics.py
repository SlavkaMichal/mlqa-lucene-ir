from sklearn.metrics import f1_score
from .retrieval import Searcher
from .translator import Translator
from .reader import Reader
from torch.utils.data import Dataset, DataLoader
from .utils import get_root
from .datasets import MLQADataset, Wiki
import random
import os
import pdb
import numpy as np


def hits(dataset, langContext, langQuestion, distant=False, saveas=None, k=50):
    searcher = Searcher()
    searcher.addLang(
        lang=langContext,
        analyzer=langContext,
        dataset=dataset)

    data = MLQADataset(dataset, langContext, langQuestion)

    # file to save metrics
    root = get_root()
    metric = "hitAtk_"
    if dist:
        metric = "dist_"
    if saveas == None:
        saveas = os.path.join(root,"data/stats/{}{}-C{}-Q{}"
                .format(metric, dataset, langContext, langQuestion))
    else:
        saveas = os.path.join(root,"data/stats/{}".format(saveas))
    print("Saving stats as {}".format(saveas))

    # counters
    dtype = np.dtype([('total', 'i4'), ('hits', 'i8',(k)), ('scores', 'f8', (k))])
    misses = []
    tally = np.zeros(1, dtype=dtype)[0]
    for doc in data.get():
        tally['total'] += 1
        # TODO it looks like the list is ordered by score
        # but should not be trusted
        scoreDocs = self.query(doc['question'], k)
        hit = False
        for n, scoreDoc in enumerate(scoreDocs):
            ret_doc = searcher.getDoc(scoreDoc.doc)
            # list of qa ids
            docIds = [ id.stringValue() for id in ret_doc.getFields('id')]
            # check if the document is a hit
            if doc['qid'] in docIds or (distant and doc['answer'] in ret_doc.get('context')):
                tally['hits'][n] += 1
                tally['scores'][n] += scoreDoc.score
                hit = True
                break
        if not hit:
            misses.append({'question':qa['question'],
                'context' : paragraph['context']})

    with open(saveas+"-misses.json", "w+") as fp:
        json.dump(misses, fp)
    np.save(saveas+".npy", tally)
    print("Evaluation of retrieval done")
    return

def qa_f1(dataset, eval_dataset, langSearch, langQuestion, saveas=None, k=50):
    searcher = Searcher()
    searcher.addLang(
        lang=langSearch,
        analyzer=langSearch,
        dataset=dataset)

    tr_langs = {'en', langSearch, langQuestion}
    translator = Translator(tr_langs)

    data = MLQADataset(eval_dataset, langSearch, langQuestion)

    reader = Reader()
    reader.addSearcher(searcher, k)
    reader.addTranslator(translator, tr_langs)
    # file to save metrics
    root = get_root()
    metric = "qa_f1_"
    if saveas == None:
        saveas = os.path.join(root,"data/stats/{}{}-C{}-Q{}"
                .format(metric, dataset, langSearch, langQuestion))
    else:
        saveas = os.path.join(root,"data/stats/{}".format(saveas))
    print("Saving stats as {}".format(saveas))

    # counters
    dtype = np.dtype([('hits', 'i8'),('exact', 'i8'),
        ('total', 'i8'), ('f1', 'f8'), ('score', 'f8')])
    misses = []
    tally = np.zeros(1, dtype=dtype)[0]
    try:
        for n, doc in enumerate(data.get()):
            #print(tally['total'],doc['title'])
            # Question is in langQuestion need to be translated to langSearch
            result  = reader.answer(doc['question'], langQuestion, langSearch)
            #tok_answer_model = reader.tokenizer(result['answer'])['input_ids']
            #tok_answer_gold = reader.tokenizer(doc['answer'])['input_ids']
            # TODO it looks like the list is ordered by score
            # but should not be trusted
            #tally['f1'] += f1_score(tok_answer_gold, tok_answer_model)
            tally['f1'] += f1_score(doc['answer'], result['answerSearch'])
            tally['total'] += 1
            tally['score'] += result['score_read']
            #tally['hits'] += doc['qid'] in [ id.stringValue() for id in result['doc'].getFields('id')]
            tally['hits'] += int(doc['answer'] in result['contextSearch'])

            if (n%10000 == 0):
                print(n)

            #for k, v in result.items():
            #    if type(v) == str:
            #        v = v[:35]
            #    print("{:<15}: {}".format(k,v))
            #print("-"*55)
            #if n > 3:
            #    print("breaking")
            #    break
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        print("Dataset: {}".format(dataset))
        print("Context: {}".format(langSearch))
        print("Question: {}".format(langQuestion))
        if tally['total'] > 0:
            print("F1: {}".format(tally['f1']/tally['total']))
            print("Total: {} questions".format(tally['total']))
            print("Hits: {}".format(tally['hits']))
            print("Exact matches: {}".format(tally['exact']))
            print("Mean score: {}".format(tally['score']/tally['total']))
            print("Evaluation of retrieval done")
        np.save(saveas+".npy", tally)
    return

def f1_score(gt, prediction):
    same_tokens = sum([token in gt for token in prediction])
    if same_tokens == 0 or len(prediction) == 0 or len(gt) == 0:
        return 0
    precision   = same_tokens / len(prediction)
    recall      = same_tokens / len(gt)
    return 2*precision*recall/(precision+recall)

def review(dataset, langContext, langQuestion, k=10):
    searcher = Searcher()
    searcher.addLang(
        lang=langContext,
        analyzer=langContext,
        dataset=dataset)

    data = MLQADataset('test', langContext, langQuestion)

    reader = Reader()
    reader.addSearcher(searcher, k)
    # counters
    tally = 0
    #lst = data.get()
    #random.shuffle(lst)
    for doc in sorted(data.get(), key=lambda k: random.random()):
        tally += 1
        print("Doc: ",tally)
        res = reader.answer(doc['question'])
        if doc['answer'] != res['answer']:
            print("Question: ",doc['question'])
            print("Answer gold: ",doc['answer'])
            print("Answer: ",res['answer'])
            print("Score: ",res['score'])
            while True:
                try:
                    command = input("Command: ")
                except EOFError:
                    print("Exiting")
                    return
                if command == "":
                    continue
                if command == 'q':
                    print("Question: ",doc['question'])
                elif command == 'next':
                    break
                elif command == 'cg':
                    print("Context gold: ",doc['context'])
                elif command == 'ac':
                    tmp = reader(doc['question'],doc['context'])
                    print("Answer from correct context: ",tmp[1])
                elif command == 'ag':
                    print("Answer gold: ",doc['answer'])
                elif command == 'a':
                    print("Answers: ",res['answers'])
                elif command == 'c':
                    print("Context result: ",res['doc'].get("context"))
                elif command == 'sret':
                    print("Score retriever: ",res['ret_score'])
                elif command == 's':
                    print("Scores: ",res['scores'])
                elif command == 'pos':
                    print("Position: ",res['pos'])
                elif command == 'n':
                    print("Best reader score: ",res['n'])
                elif command == 'u':
                    print("Used document: ")
                    searcher.printDoc(res['doc'])
                elif command == 'pret':
                    scoreDocs = searcher.query(doc['question'],n=3)
                    searcher.printResult(scoreDocs)
                elif command == 'k':
                    try:
                        num = int(input("Number of documents: "))
                    except:
                        continue
                    scoreDocs = searcher.query(doc['question'],n=num)
                    searcher.printResult(scoreDocs)
                else:
                    print("Commands: q,c,cg,a,ag,ac,s,sret,n,u,pret,k")
                    print("Press C^D for exit")
                    print("Write 'next' to continue")
    return


###############################################################################
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
