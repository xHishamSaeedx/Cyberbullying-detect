import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.util import pr
stemmer = nltk.SnowballStemmer('english')
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words("english"))

df = pd.read_csv("twitter_data.csv")
df['labels'] = df['class'].map({0:"Hate Speech Detected" , 1:"Offensive language detected" , 2: "No hate and offensive speech"})
df = df[['tweet' , 'labels']]

def clean(text):
  text = str(text).lower()
  text = re.sub('\[.*?\]' , '', text)
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+' , '' , text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '' , text)
  text = re.sub('\n' , '' , text)
  text = re.sub('\w*\d\w*', ''  , text)
  text = [word for word in text.split(' ') if word not in stopword]
  text = " ".join(text)
  text = [stemmer.stem(word) for word in text.split(' ')]
  text = " ".join(text)
  return text

df["tweet"] = df["tweet"].apply(clean)

x = np.array(df["tweet"])
y = np.array(df["labels"])

cv = CountVectorizer()
x = cv.fit_transform(x)

def randomforestpredict(model , text):
  test_data = text
  df = cv.transform([test_data]).toarray()
  output = model.predict(df)
  print(output)
  probabilities = model.predict_proba(df)
  print(f"randomforest : {probabilities}")
  return probabilities[0]

