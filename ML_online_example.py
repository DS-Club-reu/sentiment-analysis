from online_models.ML_online import SentimentML

""" 
Если pickle файл с обученной моделью НЕ лежит в папке с вашим py файлом, укажите путь 
                                                                        в переменную path 
"""
path = 'saved_models/Sentiment_model.pickle'

# В список text_to_predict необходимо добавить все тексты, которые вы хотите проверить на тональность
texts_to_predict = 'Любовь'

extrasensory = SentimentML()
extrasensory.load_ml(path)

print(extrasensory.predict(texts_to_predict))