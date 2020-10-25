from sklearn.metrics import f1_score
from retrieval import Searcher

def hitAtK(data, dataset, langContext, langQuestion, saveas=None, k=50):
    dataname = self.dataname(dataset, langContext, langQuestion)
    data = data['mlqa_'+dataset][dataname]['data']

    root = get_root()
    if saveas == None:
        saveas = os.path.join(root,"data/stats/{}-C{}-Q{}"
                .format(dataset, langContext, langQuestion))
    else:
        saveas = os.path.join(root,"data/stats/{}".format(saveas))
    print("Saving stats as {}".format(saveas))
    # counters
    dtype = np.dtype([('total', 'i4'), ('hits', 'i8',(k)), ('scores', 'f8', (k))])
    misses = []
    tally = np.zeros(1, dtype=dtype)[0]
    for doc in data:
        #title = doc['title']
        for paragraph in doc['paragraphs']:
            for qa in paragraph['qas']:
                tally['total'] += 1
                # TODO it looks like the list is ordered by score
                # but should not be trusted
                scoreDocs = self.query(qa['question'], k)
                prevScore = math.inf
                qid = qa['id']
                hit = False
                for n, scoreDoc in enumerate(scoreDocs):
                    doc = self.searcher.doc(scoreDoc.doc)
                    docIds = [ id.stringValue() for id in doc.getFields('id')]
                    # sanity check
                    # TODO remove condition if it works
                    if prevScore < scoreDoc.score:
                        print("Previous score: {} was smaller than current score: {}".
                                format(prevScore, scoreDoc.score))
                        self.printResult(scoreDocs)
                        return
                    prevScore = scoreDoc.score
                    # check if the document is a hit
                    if qid in docIds:
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

