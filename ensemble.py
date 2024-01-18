from lstmtest3 import lstm_predict

from tensorflow.keras.models import load_model


#LSTM MODEL
lstm_model = load_model('model3.h5')


#BERT MODEL 

# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Use a pipeline as a high-level helper
from transformers import pipeline


classes = {0 : "hate speech" , 1 : "normal" , 2 : "offensive language" }

tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
pipe = pipeline("text-classification", model="Hate-speech-CNERG/bert-base-uncased-hatexplain")


def bert_predict(text):
    output = pipe(text)
    print(output)
    if output[0]['label'] == 'normal':
        return 0
    else: 
        return 1

##RANDOM FOREST
import joblib

model_filename = 'random_forest_model.pkl'
randomforest_model = joblib.load(model_filename)

from randomforesttest import randomforestpredict


total_output = []
total_output.append(lstm_predict(lstm_model , "fucking asshole nigger"))
total_output.append(bert_predict("fucking asshole nigger"))
total_output.append(randomforestpredict(randomforest_model , "fucking asshole nigger"))

print(total_output)