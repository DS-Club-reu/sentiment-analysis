import telebot
import pickle
from online_models.ML_online import SentimentML
from online_models.DL_online import SentimentDL

model = SentimentML()
model.load_ml('Saved_models/Sentiment_model.pickle')

# model = SentimentDL()
# model.load_model('Saved_models/Bert_weights.pth')

instruction = """Чтобы начать анализ текстов, напишите 'Cтарт'.

Далее присылайте тексты для анализа на русском языке. Бот будет отвечать вам, позитивные это тексты или негативные.

Чтобы завершить тренировку, напишите 'Конец'."""

bot = telebot.TeleBot('5014633189:AAFeB_5Ax7-pW13iKcEcuqi7uS9Axy5F60Q')

@bot.message_handler(commands=['start'])
def start_command(message):
    bot.send_message(message.chat.id, "Добро пожаловать в бот проекта по предсказанию тональностей текстов. Поиграем?)")

active = False

active_dict = {}

@bot.message_handler(content_types=['text', 'document', 'audio'])
def send_feedback(message):
    global active_dict
    id_user = message.chat.id

    if id_user not in active_dict.keys():
        active_dict[id_user] = False

    active = active_dict[id_user]

    text_lower = message.text.lower()

    if text_lower == "старт":

        active_dict[id_user] = True
        text = """Начнем игру. Пришлите мне тексты, чтобы я их проанализировал."""
        bot.send_message(message.from_user.id, text)

    elif text_lower == "конец":

        active_dict[id_user] = False
        text = """Было здорово. Надеюсь, вам понравилось! До встречи в следующий раз"""
        bot.send_message(message.from_user.id, text)

    elif message.text == "/help":
        bot.send_message(message.from_user.id, instruction)

    else:
        if active_dict[id_user] :
            prediction = model.predict(text_lower)
            bot.send_message(message.from_user.id, prediction)
        else:

            text = """Хммм. Похоже, бот выключен. Напишите 'старт' для начала анализа тестов.    
                    Если у вас есть вопросы, напишите /help."""

            bot.send_message(message.from_user.id, text)


bot.polling(none_stop=True, interval=0)