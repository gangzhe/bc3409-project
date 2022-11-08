from tkinter import *
import os, time
import joblib
import pandas as pd
import numpy as np
import nltk, string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sentence_transformers import SentenceTransformer


MODEL_FILENAME = "model.pkl"
model = joblib.load(MODEL_FILENAME)
bert = SentenceTransformer('bert-base-nli-mean-tokens')

def preprocess_tweet(tweet):
    stop_words = stopwords.words('english')
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()

    # lowercase
    tweet = tweet.lower()
    # remove punctuation
    tweet = ''.join([c for c in tweet if c not in string.punctuation])
    # remove stopwords and apply stemming
    tweet = ' '.join([wnl.lemmatize(w) for w in tweet.split() if w not in stop_words])
    
    return tweet


def analyze_tweet():
    global model
    tweet = text_tweet_analyzer.get("1.0", "end-1c")
    tweet = preprocess_tweet(tweet)

    X = bert.encode(tweet).reshape(1, -1)
    score = model.predict_proba(X)
    sentiment = 'Bull' if (int(model.predict(X)) == 1) else 'Bear'

    print(score)
    label_tweet_analyzer_result['text'] = f"{score[0][np.argmax(score[0])]*100:.2f}% {sentiment}"


def scrape_latest_tweets():   
    os.system('start cmd /k "python scraper.py"')

    while True:
        try:
            df = pd.read_csv('latest_tweets.csv')
            break
        except FileNotFoundError:
            print('sleep')
            time.sleep(5)

    df["preprocessed_tweet"] = df["Cleaned Tweet"].apply(preprocess_tweet)
    X = bert.encode(df["preprocessed_tweet"])
    mean_score = np.mean(model.predict_proba(X))
    mean_score = np.mean(model.predict(X))
    sentiment = 'Bull' if mean_score >= 0.5 else 'Bear'

    label_latest_result['text'] = f'{mean_score*100:.2f}% bullish => {sentiment} Market'

    

root = Tk()
root.title('Stock Tweets Analyzer GUI')
root.geometry('500x500')
root.resizable(False, False)

label_tweet_analyzer = Label(root, text="Enter tweet to analyze").place(x=200, y=25)
text_tweet_analyzer = Text(root, height=3, width=50)
text_tweet_analyzer.place(x=50, y=50)
button_tweet_analyzer = Button(root, text='Analyze Tweet', command=lambda:analyze_tweet()).place(x=205, y=110)
label_tweet_analyzer_result = Label(root)
label_tweet_analyzer_result.place(x=220, y=145)

label_latest = Label(root, text='Analyse latest tweets (will take some time)').place(x=125, y=225)
button_latest = Button(root, text='Analyse', command=scrape_latest_tweets).place(x=210, y=250)
label_latest_result = Label(root)
label_latest_result.place(x=150, y=285)

root.mainloop()