import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_checkpoint = "Mprimus/T5-summarize"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

prefix = "summarize: "

def summarize(text, max_length=128):
    text = prefix + text
    
    model_inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(model_inputs, num_beams=2, max_length=max_length)
    res = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return res[0]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--text', type=str)
    args = p.parse_args()
    text = args.text
    print(summarize(text))