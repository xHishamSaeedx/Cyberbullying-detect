import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def bert_predict(text):

    classes = {0: "hate speech", 1: "normal", 2: "offensive language"}

    tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
    model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")

    texts = [text]

    # Tokenize and format input
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Convert tensor to list
    probs_list = probs.squeeze().tolist()

    # Create a dictionary of class probabilities
    class_probabilities = {classes[i]: probs_list[i] * 100 for i in range(len(probs_list))}

    print("Text:", texts[0])
    print("Class Probabilities:", class_probabilities)

    return(list(class_probabilities.values()))
