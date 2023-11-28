import os
import telebot
from newspaper import Article
import nltk
from nltk import sent_tokenize
nltk.download('punkt', quiet=True)

from summarize.predict import summarize
from translate.predict import translate

handle_text = True
max_tokens = 300

print("Telegram bot is running...")
bot = telebot.TeleBot(os.getenv("TELEGRAM_TOKEN"))

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Welcome to Translate and Summarize bot!')


@bot.message_handler(commands=['text'])
def handle_text(message):
    global handle_text
    handle_text = True
    bot.send_message(message.chat.id, "Please send me text, please")

@bot.message_handler(commands=['link'])
def handle_link(message):
    global handle_text
    handle_text = False
    bot.send_message(message.chat.id, "Please send me the link of the website, please")

@bot.message_handler(content_types=['text'])
def get_text(message):
    global handle_text
    
    text = message.text
    if not handle_text:
        text = get_text_from_url(text)
    
    text = split_large_text(text)
    
    text = [translate_and_summarize(p) for p in text]
    text = ' '.join(text)
    
    response = f"Translated and summarized text: \n\n{text}"
    bot.send_message(message.chat.id, response)


def translate_and_summarize(text: str) -> str:
    return translate(summarize(text))


def get_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    
    return article.text


def split_large_text(text):
    global max_tokens
    
    paragraphs = []
    num_tokens = 0
    cur_p = []
    tokens_array = []
    for sent in sent_tokenize(text):
        tokens = len(sent.split(' '))
        if tokens == 0:
            continue
        
        num_tokens += tokens
        if num_tokens > max_tokens:
            paragraphs.append((tokens_array, cur_p))
            cur_p = []
            num_tokens = tokens
            tokens_array = []
        
        cur_p.append(sent)
        tokens_array.append(num_tokens)
    
    if len(cur_p) > 0:
        paragraphs.append((tokens_array, cur_p))
    
    if len(paragraphs) > 1:
        # balance last two paragraphs
        while len(paragraphs[-2][1]) > 1:
            ta1, par1 = paragraphs[-2]
            ta2, par2 = paragraphs[-1]
            
            if ta2[-1] > ta1[-1]:
                break
            
            par2.insert(0, par1.pop())
            ta1.pop()
            
    
    paragraphs = [' '.join(p[1]) for p in paragraphs]
    
    return paragraphs


bot.infinity_polling()