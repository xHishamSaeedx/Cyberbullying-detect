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
    print(pipe(text))



# inputs = tokenizer("just die asians" , return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

# print(f"{predicted_class_ids} ok")

# predicted_class_ids = predicted_class_ids.numpy()

# print(classes[predicted_class_ids[0]])