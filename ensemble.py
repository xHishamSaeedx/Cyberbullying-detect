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

total_output = predict_outputs("i hope you fucking die")
print(total_output)

def adjust_normal_percentage(normal_percentage, hate_percentage, offensive_percentage, min_normal_percentage=50):
    # Calculate the combined percentage of "Hate" and "Offensive"
    combined_percentage = hate_percentage + offensive_percentage

    # Check if "Normal" has a higher percentage than each of the other classes
    normal_has_priority = normal_percentage > hate_percentage and normal_percentage > offensive_percentage

    # Adjust "Normal" percentage based on priority condition
    if normal_has_priority and normal_percentage >= min_normal_percentage:
        adjusted_normal_percentage = normal_percentage
    elif normal_has_priority and normal_percentage < min_normal_percentage:
        adjusted_normal_percentage = min_normal_percentage
    else:
        adjusted_normal_percentage = combined_percentage / 2

    return adjusted_normal_percentage

def dynamic_threshold_prediction(bert_confidence, lstm_confidence, rf_confidence):
    # Define weights based on priority order
    bert_weight = 0.6  # Increased weight for BERT
    lstm_weight = 0.3  # Decreased weight for LSTM
    rf_weight = 0.1

    total_normal = (bert_weight*bert_confidence) + (lstm_weight*lstm_confidence) + (rf_weight*rf_confidence)

    if total_normal >= 0.5:
        print("Normal")
    else:
        print("Cyberbullying")



# Example usage
bert_confidence = total_output[1]

bert_confidence = adjust_normal_percentage(bert_confidence[1], bert_confidence[0] , bert_confidence[2])
lstm_confidence = total_output[0]  # LSTM has Hate and Normal only
rf_confidence = total_output[2]
rf_confidence = adjust_normal_percentage(rf_confidence[2],rf_confidence[0] , rf_confidence[1])



dynamic_threshold_prediction(bert_confidence//100, lstm_confidence[1], rf_confidence)
