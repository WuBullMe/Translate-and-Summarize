from transformers import pipeline

# model_names in huggingface
model_names = {
    'en_ru': 'Helsinki-NLP/opus-mt-en-ru',
    'ru_en': 'Helsinki-NLP/opus-mt-ru-en',
}

class Translator:
    def __init__(self, src_tg_lan='en_ru'):
        if src_tg_lan not in model_names:
            raise ValueError(f"incorrect src_tg_lan name or not supported, valid values {list(model_names.keys())}")
        
        self.model_name = model_names[src_tg_lan]
        self.pipe = pipeline("translation", model=self.model_name)
        
    def __call__(self, text):
        _text = self.pipe(text)
        
        return _text[0]['translation_text']
            