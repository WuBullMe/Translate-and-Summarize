from transformers import pipeline

# model_names in huggingface
model_names = {
    'en': 'facebook/bart-large-cnn',
    'ru': 'IlyaGusev/mbart_ru_sum_gazeta',
}

class Summarizer:
    def __init__(self, src_lan='en', max_len=140):
        if src_lan not in model_names:
            raise ValueError(f"incorrect src_lan name or not supported, valid values {list(model_names.keys())}")

        self.max_len = max_len
        self.model_name = model_names[src_lan]
        self.pipe = pipeline("summarization", model=self.model_name)
        
    def __call__(self, text):
        _text = self.pipe(text)
        return _text[0]['summary_text']