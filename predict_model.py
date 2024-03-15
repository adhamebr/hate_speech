# python3 ./nlp_assignment-master/predict_model_bert.py
import torch
from transformers import BertTokenizer
import pandas as pd
import torch
import os

from preprocessing import my_pipeline

def calculate_accuracy_for_class(labels, pred_labels, class_label):
    # Filter true labels and predicted labels for the specified class
    true_labels_for_class = [true_label for true_label, pred_label in zip(labels, pred_labels) if true_label == class_label]
    pred_labels_for_class = [pred_label for true_label, pred_label in zip(labels, pred_labels) if true_label == class_label]
    
    # Calculate accuracy for the specified class
    accuracy_for_class = sum(1 for true, pred in zip(true_labels_for_class, pred_labels_for_class) if true == pred) / len(true_labels_for_class)
 
    
    return accuracy_for_class
#os.chdir('./nlp_assignment-master/')
model_path = './models/bert_model.pt' 
model = torch.load(model_path)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
test_data_path = './data/hate-speech-dataset/test_data.csv'

df = pd.read_csv(test_data_path)
# Assuming your CSV has columns 'text' for the input and 'label' for the labels
texts = df['text']
labels = df['label_id']

pred_labels = []
for text in texts:
    clean_text =my_pipeline(text)
    with torch.no_grad():
        logits = model(**clean_text)[0]
        probabilities = torch.softmax(logits, dim=1)
        # Get predicted class label
        predicted_class = torch.argmax(probabilities, dim=1).item()
        pred_labels.append(predicted_class)
        # Write the prediction to a file


accuracy_for_class_0 = calculate_accuracy_for_class(labels, pred_labels, class_label=0)
accuracy_for_class_1 = calculate_accuracy_for_class(labels, pred_labels, class_label=1)
correct = sum(1 for true, pred in zip(labels, pred_labels) if true == pred)
accuracy = correct / len(labels)

print("Accuracy for class hate:", accuracy_for_class_0)
print("Accuracy for class no hate:", accuracy_for_class_1)
print('Test accuracy = ',accuracy)

