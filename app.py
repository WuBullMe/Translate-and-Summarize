import os
import telebot

from summarize.predict import summarize
from translate.predict import translate

print("Telegram bot is running...")
bot = telebot.TeleBot(os.getenv("TELEGRAM_TOKEN"))

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Welcome to Translate and Summarize bot!')


@bot.message_handler(commands=['text'])
def transfer_style(message):
    bot.send_message(message.chat.id, "Please send me text, please")


@bot.message_handler(content_types=['text'])
def get_text(message):
    text = translate_and_summarize(message.text)
    
    response = f"Translated and summarized text: \n\n{text}"
    bot.send_message(message.chat.id, response)


def translate_and_summarize(text: str) -> str:
    return translate(summarize(text))


bot.infinity_polling()