import string
import pymorphy2
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


class SentimentML:
    """
    A class used to build an online sentiment analysis model
    ...
    Attributes:
**************************************************************************************************************
    texts_to_predict text for predictions; comes from the user
    extrasensory - initializer of the class
    xgb is an attribute that stores the loaded trained model in the pickle format
    stop_words - saves all stop words from the library for further deletion of these stop words
                 from the dataframe
    lemma is an attribute required to normalize words\texts in a dataframe;
    spec_chars - saves all punctuation characters from the library for their further
                     deletions from a dataframe
**************************************************************************************************************

    Methods:
**************************************************************************************************************
    def preprocessing(self, df):
        Input data preprocessing function
    def training_model(self, df):
        The function divides the data into training and test samples and trains a machine learning model
    def predict(self, texts_to_predict):
        The function predicts the polarity of the input text
    def save_ml(self):
        The function saves the trained model in pickle format
    def load_ml(self, model_path):
        The function loads a file in pickle format with a trained model
**************************************************************************************************************
    """

    def __init__(self, model=XGBClassifier(base_score=0.5, learning_rate=0.9, max_depth=9, n_estimators=500,
                                   random_state=50),
                 lemmatizer=pymorphy2.MorphAnalyzer(),
                 stop_words=stopwords.words('russian'),
                 spec_chars=string.punctuation + string.digits + '\n\xa0«»\t-—–“”→...'):
        """ It is assumed that the user has a file with the model in the same folder with the py file """

        self.model = model
        self.stop_words = stop_words
        self.spec_chars = spec_chars
        self.lemmatizer = lemmatizer

    def preprocessing(self, text):
        """ Input data preprocessing function. The method returns the processed text data in list format

        Parameters:
             df : pd.DataFrame
        """

        if type(text) == type('str'):
            stop_free = " ".join([i for i in text.lower().split() if i not in self.stop_words])
            punc_free = ''.join(ch for ch in stop_free if ch not in self.spec_chars)
            lem_text = " ".join([self.lemmatizer.parse(word)[0].normal_form for word in punc_free.split(' ')])
            return lem_text
        else:
            raise TypeError('Text to preprocessing should be in sting format')

    def training_model(self, data):
        """ The method divides the data into training and test samples and trains a machine learning model.
            The method returns the trained model

            Parameters:
             df : pd.DataFrame

             ПРОПИСАТЬ, ЧТО НУЖЕН СЕНТИМЕНТ
        """
        df = data
        new_texts = []
        texts = df['text'].values.tolist()
        for i in range(len(texts)):
            text = texts[i]
            lem_text = self.preprocessing(text)
            new_texts.append(lem_text)

        # Preprocessing and prediction
        X = new_texts
        y = data['sentiment']

        # Split into train_test split
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

        # Modelling
        pipeline = Pipeline([('TF-IDF', TfidfVectorizer(min_df=0.01, max_df=0.9)),
                             ('ML_model', self.model)])

        model_train = pipeline
        model_train.fit(X_train, y_train)

        self.model = model_train
        return model_train

    def predict(self, texts_to_predict):
        """ The method predicts the polarity of the input text.
            The method returns an estimate of the polarity of the text

        Parameters:
             texts_to_predict : the text coming from the user to check his assessment of sentiment
        """

        texts = texts_to_predict
        prepr_tockens = [self.preprocessing(texts)]
        label_pred = self.model.predict(prepr_tockens)

        if label_pred[0] == 1:
            return 'Positive'
        else:
            return 'Negative'


    def save_ml(self, path):
        """ The method saves the trained model in pickle format """

        with open(path, 'wb') as file:
            pickle.dump(self.model, file)

    def load_ml(self, path):
        """ The method loads a file in pickle format with a trained model

        Parameters:
             path : The path specified by the user
        """

        with open(path, 'rb') as file:
            self.model = pickle.load(file)
