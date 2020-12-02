from transformers import AutoTokenizer, AutoModelForQuestionAnswering
#from .retrieval import Searcher
from .translator import Translator
from torch.nn import functional as f
import torch
import pdb
import math

class Reader(object):
    def __init__(self, model=None, tokenizer=None):
        if model == None:
            model = "distilbert-base-uncased-distilled-squad"
        if tokenizer==None:
            tokenizer = "distilbert-base-uncased-distilled-squad"

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model).to(self.device)
        self.call = self.__call__
        self.searcher = None
        self.tr = None

    def __call__(self, question, context):
        """ Invoking reader """
        inp = self.tokenizer(question, context,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=512,
                truncation=True
                ).to(self.device)
        with torch.no_grad():
            starts, ends = self.model(**inp)

        # nicer method of getting the best span
        #Ps = f.softmax(starts)
        #Pe = f.softmax(ends)
        #span = torch.argmax(torch.triu(torc.matmul(Ps.T, Pe)))
        #start = span // len(Ps)
        #end   = span %  len(Ps)
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


    def answerEn(self, question):
        if self.searcher == None:
            raise RuntimeError("Searcher not initialised")
        scoreDocs = self.searcher.query(question, 'en', n=self.n)

        result = {'scores':[],'answers':[]}
        max_score=-math.inf
        #result['scores'] = []
        #result['answers'] = []
        for n, scoreDoc in enumerate(scoreDocs):
            doc = self.searcher.getDoc(scoreDoc)
            pos, answer, score = self.call(question, doc.get("context"))
            result['scores'].append(score)
            result['answers'].append(answer)
            if score > max_score:
                max_score = score
                result['answer'] = answer
                result['score'] = score
                result['doc'] = doc
                result['pos'] = pos
                result['n']   = n
                result['ret_score']  = scoreDoc.score
        return result

    def answer(self, question, langQuestion, langSearch):
        if self.searcher == None or self.tr == None:
            raise RuntimeError("Searcher not initialised")

        questionSearch = self.tr(question, langQuestion, langSearch)
        scoreDocs = self.searcher.query(questionSearch, langSearch, n=self.n)

        result = {'scores':[],'answers':[]}
        max_score=-math.inf
        #result['scores'] = []
        #result['answers'] = []
        questionRead = self.tr(question, langQuestion, 'en')
        for n, scoreDoc in enumerate(scoreDocs):
            doc = self.searcher.getDoc(scoreDoc)
            contextSearch = doc.get("context")
            contextRead = self.tr(contextSearch, langSearch, 'en')
            pos, answer, score = self.call(questionRead, contextRead)
            if score > max_score:
                max_score = score
                result['answerQuestion'] = self.tr(answer, 'en', langQuestion)
                result['answerSearch']   = self.tr(answer, 'en', langSearch)
                result['answerRead']     = answer
                result['questionSearch'] = questionSearch
                result['questionRead']   = questionRead
                result['contextQuestion'] = self.tr(contextSearch, langSearch, langQuestion)
                result['contextSearch'] = contextSearch
                result['contextRead'] = contextRead
                result['score_read'] = score
                result['score_retr']  = scoreDoc.score
                result['pos'] = pos
                result['n']   = n
        return result


    def addSearcher(self, searcher, n=50):
        self.searcher = searcher
        self.n = n

    def addTranslator(self, translator, tr_langs):
        self.tr = translator

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
