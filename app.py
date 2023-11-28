import os
import telebot
from newspaper import Article

from summarize.predict import summarize
from translate.predict import translate

handle_text = True

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
    
    original_text = ''
    text = message.text
    if not handle_text:
        original_text, text = get_text_from_url(text)
    
    text = translate_and_summarize(text)
    
    response = f"Translated and summarized text: \n\n{text}"
    bot.send_message(message.chat.id, response)


def translate_and_summarize(text: str) -> str:
    return translate(summarize(text))


def get_text_from_url(url):
    article = Article(url)
    article.download()

    article.parse()
    article.nlp()
    
    text = article.text
    summary = article.summary
    
    if len(text) > 2048:
        text = text[:2048] + "..."
    
    return text, summary

bot.infinity_polling()