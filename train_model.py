# python3 ./nlp_assignment-master/train_model.py

import os
# import pandas as pd
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
import datasets
from transformers import BertConfig, AutoModelForSequenceClassification,BertTokenizerFast, EarlyStoppingCallback
from datasets import load_dataset
import wandb


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized['label'] = examples['label_id']
    return tokenized

os.chdir('./nlp_assignment-master/')
wandb.init(project='g42-task_bert_base' )

model_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_dataset = load_dataset('csv', data_files={'train': './data/hate-speech-dataset/train_data.csv'})
train_dataset = train_dataset['train'].shuffle(seed=42)
test_dataset = load_dataset('csv', data_files={'train': './data/hate-speech-dataset/test_data.csv'})
test_dataset = test_dataset['train']
# train_dataset = load_dataset('nlp_assignment-master/data/hate-speech-dataset/train_data.csv', type='csv')
# test_dataset = load_dataset('nlp_assignment-master/data/hate-speech-dataset/test_data.csv', type='csv')

# Preprocess training and validation data
train_dataset = train_dataset.map(tokenize_function,fn_kwargs={"tokenizer":tokenizer}, batched=True)
test_dataset = test_dataset.map(tokenize_function,fn_kwargs={"tokenizer":tokenizer}, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

metric = evaluate.load("accuracy")


training_arguments = TrainingArguments(
    output_dir='models/',
    logging_dir='logs',
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    evaluation_strategy='steps',
    eval_steps=500,
    logging_steps=500,
    save_steps=500,
    learning_rate=2e-5,
    num_train_epochs=20,
    warmup_ratio=0.1,
    lr_scheduler_type='linear',
    report_to='all',
    logging_first_step=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
)

  # Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("models/best_base_20")  # Adjust save directory
eval_results = trainer.evaluate(test_dataset)
print("Evaluation Results:", eval_results)

model_path ='./models/best_base_20/pytorch_model.bin'
# Assuming you're using the 'bert-base-uncased' architecture. Adjust as needed.
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', state_dict=torch.load(model_path))
torch.save(model, './models/bert_model.pt')
