# src/model_training.py

import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def preprocess_data(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    # Load dataset
    dataset = load_dataset('imdb')
    
    # Load tokenizer and preprocess data
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_dataset = dataset.map(lambda examples: preprocess_data(examples, tokenizer), batched=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./models',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch'
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['test'],
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained('./models')
    tokenizer.save_pretrained('./models')

if __name__ == '__main__':
    train_model()
