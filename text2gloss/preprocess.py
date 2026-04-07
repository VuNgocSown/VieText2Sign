import pandas as pd
import unicodedata
import json
from pathlib import Path
from typing import Tuple, Dict, Any
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\ufeff', '')
    text = ' '.join(text.split())
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    return text


def normalize_end_punctuation(text: str, gloss: str) -> Tuple[str, str]:
    end_punctuation = {'.', '?', '!'}
    text_end = text[-1] if text and text[-1] in end_punctuation else None
    gloss_end = gloss[-1] if gloss and gloss[-1] in end_punctuation else None
    text_clean = text.rstrip('.?!')
    gloss_clean = gloss.rstrip('.?!')
    final_punct = text_end or gloss_end or '.'
    return text_clean + final_punct, gloss_clean + final_punct


def load_data(file_path: str, config: Any) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_excel(file_path, header=None, names=['text', 'gloss'])
    print(f"Loaded {len(df)} samples from {file_path}")
    
    df['text'] = df['text'].apply(preprocess_text)
    df['gloss'] = df['gloss'].apply(preprocess_text)
    
    df = df[(df['text'] != '') & (df['text'] != 'nan') & 
            (df['gloss'] != '') & (df['gloss'] != 'nan')]
    print(f"After cleaning: {len(df)} samples")
    
    def normalize_row(row: pd.Series) -> pd.Series:
        text_norm, gloss_norm = normalize_end_punctuation(row['text'], row['gloss'])
        return pd.Series({'text': text_norm, 'gloss': gloss_norm})
    
    df[['text', 'gloss']] = df.apply(normalize_row, axis=1)
    print(f"Normalized end punctuation")
    
    duplicates_count = df.duplicated(subset=['text', 'gloss']).sum()
    df = df.drop_duplicates(subset=['text', 'gloss'], keep='first')
    print(f"Removed {duplicates_count} duplicates, final size: {len(df)} samples")
    
    if len(df) > 0:
        print(f"Sample: Text='{df['text'].iloc[0][:50]}...', Gloss='{df['gloss'].iloc[0][:50]}...'")
    
    return df


def split_data(df: pd.DataFrame, config: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets.
    
    Lần đầu: split ngẫu nhiên và lưu indices ra file JSON.
    Lần sau: load đúng indices đó để đảm bảo tain/val/test luôn nhất quán.
    """
    split_cache_path = Path(config.DATA_FILE).parent / "split_indices.json"
    
    if split_cache_path.exists():
        # Load split đã có sẵn
        with open(split_cache_path, 'r') as f:
            split_indices = json.load(f)
        
        train_idx = split_indices['train']
        val_idx   = split_indices['val']
        test_idx  = split_indices['test']
        
        # Kiểm tra indices hợp lệ
        all_cached = set(train_idx) | set(val_idx) | set(test_idx)
        if max(all_cached) >= len(df):
            print("[WARNING] Cached split indices out of range (data may have changed). Re-splitting...")
            split_cache_path.unlink()
            return split_data(df, config)  # Recursive call, tạo split mới
        
        print(f"[INFO] Loaded split from cache: {split_cache_path}")
        print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
        return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    
    # Lần đầu: tạo split mới
    train_val, test = train_test_split(
        df, test_size=config.TEST_SPLIT, random_state=config.SEED, shuffle=True
    )
    val_ratio = config.VAL_SPLIT / (1 - config.TEST_SPLIT)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=config.SEED, shuffle=True
    )
    
    # Lưu indices để tái sử dụng nhất quán
    split_indices = {
        'train': train.index.tolist(),
        'val':   val.index.tolist(),
        'test':  test.index.tolist(),
    }
    with open(split_cache_path, 'w') as f:
        json.dump(split_indices, f)
    print(f"[INFO] Split indices saved to {split_cache_path}")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    return train, val, test


def tokenize_function(examples: Dict[str, list], tokenizer: Any, max_length: int, task_prefix: str = None) -> Dict[str, list]:
    inputs = examples["text"]
    if task_prefix:
        inputs = [task_prefix + text for text in inputs]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    labels = tokenizer(text_target=examples["gloss"], max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_datasets(config: Any, tokenizer: Any) -> DatasetDict:
    df = load_data(config.DATA_FILE, config)
    train_df, val_df, test_df = split_data(df, config)
    
    datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False)
    })
    
    task_prefix = getattr(config, "TASK_PREFIX", None)
    
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer, 
            "max_length": config.MAX_LENGTH,
            "task_prefix": task_prefix
        },
        remove_columns=datasets["train"].column_names
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(tokenized_datasets['train']):5d}")
    print(f"  Val:   {len(tokenized_datasets['validation']):5d}")
    print(f"  Test:  {len(tokenized_datasets['test']):5d}")
    
    return tokenized_datasets
