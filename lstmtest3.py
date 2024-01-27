import os
import re
import string

import pandas as pd
import numpy as np

from collections import Counter

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

main_data=pd.read_csv("train.csv")
data=main_data.copy()
data.drop(columns=['id'],axis=1,inplace=True)

tf.random.set_seed(0)

#Balancing the dataset using Oversampling
data1=data[data['label']==1]
data0=data[data['label']==0]
data=pd.concat([data,data1,data1], axis=0)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)


def clean_text(text ):
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>3))])

    return text2.lower()

data['tweet'] = data['tweet'].apply(remove_emoji)
data['tweet'] = data['tweet'].apply(clean_text)
data['Num_words_text'] = data['tweet'].apply(lambda x:len(str(x).split()))

train_data,test_data= train_test_split(data, test_size=0.2 , random_state = 0)
train_data.reset_index(drop=True,inplace=True)
test_data.reset_index(drop=True,inplace=True)


#train and validation dataset splitting
X_train, X_valid, y_train, y_valid = train_test_split(train_data['tweet'].tolist(),\
                                                      train_data['label'].tolist(),\
                                                      test_size=0.2,\
                                                      stratify = train_data['label'].tolist(),\
                                                      random_state=0)


num_words = 50000

tokenizer = Tokenizer(num_words=num_words,oov_token="unk")
tokenizer.fit_on_texts(X_train)


# Convert text to sequences
x_train_sequences = tokenizer.texts_to_sequences(X_train)
x_valid_sequences = tokenizer.texts_to_sequences(X_valid)
x_test_sequences = tokenizer.texts_to_sequences(test_data['tweet'].tolist())

# Pad sequences to make them of equal length
maxlen = 50  # adjust as needed
x_train_padded = pad_sequences(x_train_sequences, padding='post', maxlen=maxlen)
x_valid_padded = pad_sequences(x_valid_sequences, padding='post', maxlen=maxlen)
x_test_padded = pad_sequences(x_test_sequences, padding='post', maxlen=maxlen)

# Convert to NumPy array
x_train = np.array(x_train_padded)
x_valid = np.array(x_valid_padded)
x_test = np.array(x_test_padded)

train_labels = np.asarray(y_train)
valid_labels = np.asarray(y_valid)
test_labels = np.asarray(test_data['label'].tolist())

#tensorflow dataset preparation
train_ds = tf.data.Dataset.from_tensor_slices((x_train,train_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((x_valid,valid_labels))
test_ds = tf.data.Dataset.from_tensor_slices((x_test,test_labels))


def lstm_predict(model , text):
    data = {'id': [1],
            'tweet': [text]}

    # Create a DataFrame
    ftest = pd.DataFrame(data)

    ftest.drop(columns=['id'],axis=1,inplace=True)

    ftest['tweet'] = ftest['tweet'].apply(remove_emoji)
    ftest['tweet'] = ftest['tweet'].apply(clean_text)

    # Convert text to sequences for the test data
    f_test_sequences = tokenizer.texts_to_sequences(ftest['tweet'].tolist())

    # Pad sequences for the test data
    f_test_padded = pad_sequences(f_test_sequences, padding='post', maxlen=maxlen)

    # Convert to NumPy array
    f_test = np.array(f_test_padded)

    #predict on actual test data
    predictions = model.predict(f_test)


    # cutoff = 0.1
    ftest['pred_sentiment']= predictions
    # print(f"LSTM : {ftest}")
    # ftest['pred_sentiment'] = np.where((ftest.pred_sentiment >= cutoff),1,ftest.pred_sentiment)
    # ftest['pred_sentiment'] = np.where((ftest.pred_sentiment < cutoff),0,ftest.pred_sentiment)

    # #processed tweets categorized as hate speech
    # pd.set_option('display.max_colwidth', None)
    # ftest[ftest['pred_sentiment']==1]

    return [ftest['pred_sentiment'].iloc[0] , 1- ftest['pred_sentiment'].iloc[0]]


