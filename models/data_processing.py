# src/data_processing.py

from datasets import load_dataset
from transformers import AutoTokenizer

def download_data():
    # Load the dataset from Hugging Face Datasets
    dataset = load_dataset('emotion')
    return dataset

def preprocess_data(dataset):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Define a preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    # Apply the preprocessing function to the dataset
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    # Remove the unnecessary columns
    encoded_dataset = encoded_dataset.remove_columns(['text'])
    encoded_dataset.set_format('torch')
    
    return encoded_dataset

def save_processed_data(dataset, save_path):
    # Save the processed dataset to disk
    dataset.save_to_disk(save_path)

def main():
    # Download the data
    dataset = download_data()
    
    # Preprocess the data
    processed_dataset = preprocess_data(dataset)
    
    # Save the processed data
    save_path = './data/processed/imdb'
    save_processed_data(processed_dataset, save_path)
    print(f"Processed data saved to {save_path}")

if __name__ == '__main__':
    main()
