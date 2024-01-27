from lstmtest3 import lstm_predict

from tensorflow.keras.models import load_model


#LSTM MODEL
lstm_model = load_model('model3.h5')

from berttest2 import bert_predict

##RANDOM FOREST
import joblib

model_filename = 'random_forest_model.pkl'
randomforest_model = joblib.load(model_filename)

from randomforesttest import randomforestpredict

def predict_outputs(text):
    total_output = []
    total_output.append(lstm_predict(lstm_model , text))
    total_output.append(bert_predict(text))
    total_output.append(randomforestpredict(randomforest_model , text))
    return total_output

total_output = predict_outputs("stupid black nigger")
print(total_output)


def dynamic_threshold_prediction(bert_confidence, lstm_confidence, rf_confidence):
    # Define weights based on priority order
    bert_weight = 0.6
    lstm_weight = 0.3
    rf_weight = 0.1

    # Calculate weighted average for Cyberbullying and Normal
    weighted_avg_cyberbullying = (bert_weight * (bert_confidence[0] + bert_confidence[2]) +
                                  lstm_weight * lstm_confidence[0] +
                                  rf_weight * (rf_confidence[0] + rf_confidence[1]))

    weighted_avg_normal = (bert_weight * bert_confidence[1] +
                           lstm_weight * lstm_confidence[1] +
                           rf_weight * rf_confidence[2])

    # Set an initial threshold
    initial_threshold = 0.5

    # Dynamic threshold adjustment based on confidence
    dynamic_threshold = initial_threshold + (weighted_avg_cyberbullying - weighted_avg_normal)

    # Determine the final predicted class
    if weighted_avg_cyberbullying > dynamic_threshold:
        predicted_class = "Cyberbullying"
    else:
        predicted_class = "Normal"

    return predicted_class


# Example usage
bert_confidence = total_output[1]
lstm_confidence = total_output[0]  # LSTM has Hate and Normal only
rf_confidence = total_output[2]

prediction = dynamic_threshold_prediction(bert_confidence, lstm_confidence, rf_confidence)
print("Predicted Class:", prediction)