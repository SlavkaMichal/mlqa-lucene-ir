from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from .retrieval import Searcher
import torch
import pdb
import math

class Reader(object):
    def __init__(self, model=None):
        if model == None:
            model = "distilbert-base-uncased-distilled-squad"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.call = self.__call__
        self.searcher = None

    def __call__(self, question, context):
        inp = self.tokenizer(question, context,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=512,
                truncation=True
                )
        starts, ends = self.model(**inp)
        start1 = torch.argmax(starts)
        end1 = torch.argmax(ends[0,start1:])+start1
        end2 = torch.argmax(ends)
        start2 = torch.argmax(starts[0,:end2+1])

        score1 = starts[0,start1]+ends[0,end1]
        score2 = starts[0,start2]+ends[0,end2]
        if  score1 > score2:
            start, end, score = start1, end1+1, score1.item()
        else:
            start, end, score = start2, end2+1, score2.item()

        text = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    inp['input_ids'].tolist()[0][start:end]))
        return (start, end), text, score

    def answer(self, question):
        if self.searcher == None:
            raise RuntimeError("Searcher not initialised")
        scoreDocs = self.searcher.query(question, n=self.n)

        result = {}
        max_score=-math.inf
        result['scores'] = []
        result['answers'] = []
        for n, scoreDoc in enumerate(scoreDocs):
            doc = self.searcher.getDoc(scoreDoc)
            pos, answer, score = self.call(question, doc.get("context"))
            result['scores'].append(score)
            result['answers'].append(answer)
            if score > max_score:
                max_score = score
                result['doc'] = doc
                result['pos'] = pos
                result['n']   = n
                result['ret_score']  = scoreDoc.score
        return result

    def addSearcher(self, searcher, n=50):
        self.searcher = searcher
        self.n = n

    def run(self):
        while True:
            try:
                question = input("Question:")
            except EOFError:
                return
            if question == "":
                continue
            scoreDocs = self.searcher.query(question, n=5)
            scores_read = []
            scores_ret = []
            for scoreDoc in scoreDocs:
                doc = self.searcher.getDoc(scoreDoc)
                pos, answer, score = self.call(question, doc.get("context"))
                scores_read.append(score)
                scores_ret.append(doc.score)
                print("Question:", question)
                print("Score retrieval:", scoreDoc.score)
                print("Score reader:", score)
                print("Title:", doc.get("docname").encode('utf-8'))
                print("Context:", doc.get("context").encode('utf-8'))
                print("Answer:", answer)
                print("Position: start {}, end {}".format(pos[0],pos[1]) )
                print("----------------------------------------------------------")
            max_score = max(enumerate(scores_read),key=lambda x: x[1])
            print("Max reader score was for document {} with reader score {} and retrieval score {}"
                    .format(
                        max_score[0],
                        max_score[1],
                        scores_ret[max_score[0]]))

            print("\n\n----------------------------------------------------------")