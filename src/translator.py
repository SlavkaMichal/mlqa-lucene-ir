from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class Translator(object):

    def __init__(self, languages=['en', 'es','de']):
        lang_pairs = [ l1+"-"+l2 for l2 in languages for l1 in languages if l1 != l2 ]
        self.languages = languages
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}

        for lang_pair in lang_pairs:
            self.add_model(lang_pair)

    def __call__(self, lang, transl, sentence):
        lang_pair = lang+"-"+transl
        return self.translate(lang_pair, sentence)

    def add_model(self, lang_pair):
        model_name = "Helsinki-NLP/opus-mt-"+lang_pair
        try:
            self.models[lang_pair] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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

    def translate(self, lang, transl, sentence):
        if lang == transl:
            return sentence
        lang_pair = lang+"-"+transl
        return self.translate(lang_pair, sentence)

    def translate(self, lang_pair, sentence):
        inp = self.tokenizer[lang_pair](sentence)
        return self.model[lang_pair](inp)

