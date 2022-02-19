from online_models.DL_online import SentimentDL

Bert = SentimentDL()
Bert.load_model('Bert_weights.pth')
Bert.test("хороший телефон")