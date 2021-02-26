from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch import nn
from torch.nn import functional as F
import torch


class DistilBERT(nn.Module):
    def __init__(self, params='huggingface'):
        super().__init__()
        if params == 'huggingface':
            params = "distilbert-base-uncased-distilled-squad"
            self.model = AutoModelForQuestionAnswering.from_pretrained(params).base_model
            # independent start and end
            self.w_s = nn.Linear(768, 1)
            self.w_e = nn.Linear(768, 1)
            # joint probability
            self.W_H = nn.Linear(768, 768)
            # adjusting score from retriever score = w*score_ret+b
            self.w_ret = nn.Linear(1, 1, bias=True)
        else:
            self.load(params)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def tokenize(self, question, context):
        tokens = self.tokenizer(question, context,
                                add_special_tokens=True,
                                return_tensors='pt',
                                max_length=512,
                                truncation=True
                                )
        return tokens

    def tokenize_batch(self, question: str, contexts: list):
        batch_pairs = [(question, context) for context in contexts]
        tokens = self.tokenizer.batch_encode_plus(batch_pairs,
                                                  add_special_tokens=True,
                                                  return_tensors='pt',
                                                  max_length=512,
                                                  truncation=True
                                                  )
        return tokens

    @staticmethod
    def order_spans(starts, ends):
        batch_size, passage_length = starts.size()

        span_log_probs = torch.triu(starts.unsqueeze(2) + ends.unsqueeze(1))
        span_log_probs_sorted, indices = torch.sort(span_log_probs.view(batch_size, -1), dim=-1, descending=True)

        span_starts_indices = indices // passage_length
        span_ends_indices = indices % passage_length
        scores = F.softmax(span_log_probs_sorted, dim=-1)

        return span_starts_indices, span_ends_indices, scores

    @classmethod
    def select_span(cls, starts, ends, mode):
        if mode == 'max':
            span_starts_indices, span_ends_indices, scores = cls.order_spans(starts, ends)

            # TODO only one batch is expected at the moment
            start = span_starts_indices[0, 0]
            end = span_ends_indices[0, 0]
            score = scores[0, 0]
        elif mode == 'old':
            start1 = torch.argmax(starts)
            end1 = torch.argmax(ends[0, start1:]) + start1
            end2 = torch.argmax(ends)
            start2 = torch.argmax(starts[0, :end2 + 1])

            score1 = starts[0, start1] + ends[0, end1]
            score2 = starts[0, start2] + ends[0, end2]
            if score1 > score2:
                start, end, score = start1, end1 + 1, score1.item()
            else:
                start, end, score = start2, end2 + 1, score2.item()
        else:
            raise RuntimeError("Incorrect span selection method")

        return start, end, score

    def to_text(self, input_ids, start, end):
        text = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                input_ids.tolist()[0][start:end]))
        return text

    def forward(self, input_ids, attention_mask, retriever_score=None):
        """ Invoking reader """
        batch_size, input_size = input_ids.shape
        # hidden state
        He = self.model(input_ids, attention_mask, return_dict=True)['last_hidden_state']

        # joint probabilities
        Hs = self.W_H(He)
        starts = self.w_s(Hs)
        ends = self.w_e(He)
        joint_logs = torch.matmul(Hs, He.transpose(1, 2))
        assert joint_logs.shape == (batch_size, input_size, input_size)

        # transform retriever score
        if retriever_score is not None:
            retriever_score = self.W_ret(retriever_score)

        return starts, ends, joint_logs, retriever_score

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file: 'str'):
        self.load_state_dict(torch.load(file))

    def compute_loss(self, starts, ends, joint, retriever, answer):

        start, end, score = self.order_spans(starts, ends)

        F.cross_entropy()

        pass
