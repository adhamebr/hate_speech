# Hosts the REST API
# python3 ./nlp_assignment-master/app.py

from fastapi import FastAPI, HTTPException,Form
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import numpy as np
import pandas as pd
from starlette.responses import HTMLResponse
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os 
from preprocessing import my_pipeline

#os.chdir('./nlp_assignment-master/')
app = FastAPI()
@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input maxlength="28" name="text" type="text" value="Text to be tested" />
        <input type="submit" />'''

model_path = './models/bert_model.pt'  # Update this path to your model's location
model = torch.load(model_path)
model.eval()

@app.post('/predict')
def predict(text:str = Form(...)):
    clean_text = my_pipeline(text) #clean, and preprocess the text through pipeline
      # Predict
    with torch.no_grad():
        logits = model(**clean_text)[0]
        probabilities = torch.softmax(logits, dim=1)
        # Get predicted class label
        predicted_class = torch.argmax(probabilities, dim=1).item()
        probability = max(probabilities.tolist()[0]) #calulate the probability
       # Write the prediction to a file
    
        if predicted_class==0:
            t_sentiment = 'hate' #set appropriate sentiment
        elif predicted_class==1:
            t_sentiment = 'no hate'
        return { #return the dictionary for endpoint
            "ACTUALL SENTENCE": text,
            "PREDICTED SENTIMENT": t_sentiment,
            "Probability": probability
        }

#  then run uvicorn app:app --reload --port 8001
# then open http://127.0.0.1:8001/predict
