"""Data preprocessing for Text2Gloss"""
import pandas as pd
import re
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def load_data(file_path, config):
    """
    Load data from Excel file
    File should have 2 columns: text (column 0) and gloss (column 1)
    No header required - will be assigned automatically
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load Excel file without header
    # Assumes: Column 0 = Vietnamese text, Column 1 = Gloss
    df = pd.read_excel(file_path, header=None, names=['text', 'gloss'])
    
    # Clean data (keep punctuation as is)
    df['text'] = df['text'].astype(str).str.strip()
    df['gloss'] = df['gloss'].astype(str).str.strip()
    
    # Remove empty rows and invalid data
    df = df[(df['text'] != '') & (df['text'] != 'nan') & 
            (df['gloss'] != '') & (df['gloss'] != 'nan')]
    
    print(f"Loaded {len(df)} samples from {file_path}")
    print(f"Sample data:")
    print(f"  Text: {df['text'].iloc[0][:50]}...")
    print(f"  Gloss: {df['gloss'].iloc[0][:50]}...")
    
    return df


def split_data(df, config):
    """Split data into train/val/test sets"""
    # First split: separate test set
    train_val, test = train_test_split(
        df, test_size=config.TEST_SPLIT, random_state=config.SEED
    )
    
    # Second split: separate validation set from training
    val_ratio = config.VAL_SPLIT / (1 - config.TEST_SPLIT)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=config.SEED
    )
    
    return train, val, test


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize text and gloss"""
    # Tokenize input text
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True
    )
    
    # Tokenize target gloss
    labels = tokenizer(
        text_target=examples["gloss"],
        max_length=max_length,
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_datasets(config, tokenizer):
    """Load and prepare datasets for training"""
    # Load data
    df = load_data(config.DATA_FILE, config)
    
    # Split data
    train_df, val_df, test_df = split_data(df, config)
    
    # Create HuggingFace datasets
    datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False)
    })
    
    # Tokenize
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config.MAX_LENGTH},
        remove_columns=datasets["train"].column_names
    )
    
    # Print statistics
    print(f"\nDataset split:")
    print(f"  Train:      {len(tokenized_datasets['train']):5d} samples")
    print(f"  Validation: {len(tokenized_datasets['validation']):5d} samples")
    print(f"  Test:       {len(tokenized_datasets['test']):5d} samples")
    
    return tokenized_datasets
