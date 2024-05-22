import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import BertTokenizer

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def perform_stemming(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_words = []
    for token in tokens:
        stemmed_words.append(stemmer.stem(token))
    return ' '.join(stemmed_words)

def preProcess_data(text):
    text = text.lower()
    text = remove_stopwords(text)
    text = perform_stemming(text)
    pattern = r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

def my_pipeline(text):
    text_new = preProcess_data(text)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text_new, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs
