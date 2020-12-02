from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class Translator(object):

    def __init__(self, languages=['en', 'es','de']):
        lang_pairs = [ l1+"-"+l2 for l2 in languages for l1 in languages if l1 != l2 ]
        print("Translator language pairs: ", lang_pairs)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.languages = languages
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}

        for lang_pair in lang_pairs:
            self.add_model(lang_pair)

    def __call__(self, sentence, lang, transl):
        lang_pair = lang+"-"+transl
        if lang == transl:
            return sentence
        return self.translate(sentence, lang_pair)

    def add_model(self, lang_pair):
        model_name = "Helsinki-NLP/opus-mt-"+lang_pair
        try:
            self.models[lang_pair] = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.models[lang_pair].eval()
            self.tokenizers[lang_pair] = AutoTokenizer.from_pretrained(model_name)
            #self.pipelines[lang_pair] = pipeline(lang_pair,
            #        model=AutoModelForSeq2SeqLM.from_pretrained(model_name),
            #        tokenizer=AutoTokenizer.from_pretrained(model_name))
        except OSError as e:
            print("Failed to load nmt model for {}, probabli not suported.".
                format(lang_pair))
            raise e

    def add_language(self, new_lang):
        for lang in self.languages:
            self.add_model(new_lang+"_"+lang)
            self.add_model(lang+"_"+new_lang)
        self.languages.append(new_lang)

    def translate(self, sentence, lang, transl):
        if lang == transl:
            return sentence
        lang_pair = lang+"-"+transl
        return self.translate(sentence, lang_pair)

    def translate(self, sentence, lang_pair):
        inp = self.tokenizers[lang_pair].prepare_seq2seq_batch([sentence]).to(self.device)
        with torch.no_grad():
            out = self.models[lang_pair].generate(**inp)
        return self.tokenizers[lang_pair].decode(out[0], skip_special_tokens=True)

