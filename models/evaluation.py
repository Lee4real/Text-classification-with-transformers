# src/evaluation.py

import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
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

def evaluate_model():
    # Load dataset
    dataset = load_dataset('imdb')
    
    # Load tokenizer and preprocess data
    tokenizer = AutoTokenizer.from_pretrained('./models')
    encoded_dataset = dataset.map(lambda examples: preprocess_data(examples, tokenizer), batched=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained('./models')
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    # Evaluate model
    results = trainer.evaluate(eval_dataset=encoded_dataset['test'])

    print("Evaluation Results:", results)
    return results

if __name__ == '__main__':
    evaluate_model()
