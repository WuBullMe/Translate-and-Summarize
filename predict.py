from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse

trl_model_checkpoint = "Mprimus/T5-translation"
trl_model = AutoModelForSeq2SeqLM.from_pretrained(trl_model_checkpoint)
trl_tokenizer = AutoTokenizer.from_pretrained(trl_model_checkpoint)

sum_model_checkpoint = "Mprimus/T5-summarize"
sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_checkpoint)
sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_checkpoint)

prefix_sum = "summarize: "
prefix_trl = "translate en-ru: "

def summarize(text, max_length=128):
    text = prefix_sum + text
    
    model_inputs = sum_tokenizer.encode(text, return_tensors="pt")
    outputs = sum_model.generate(model_inputs, num_beams=2, max_length=max_length)
    res = [sum_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return res[0]

def translate(text, max_length=128):
    text = prefix_trl + text
        
    model_inputs = trl_tokenizer.encode(text, return_tensors="pt")
    outputs = trl_model.generate(model_inputs, num_beams=2, max_length=max_length)
    res = [trl_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return res[0]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--text', type=str)
    args = p.parse_args()
    text = args.text
    
    summarization = summarize(text)
    res = translate(summarization)
    print(res)
