"""Verify data split is correct"""
from . import config
from . import preprocess


def main():
    """Verify train/val/test split"""
    print("="*60)
    print("Data Split Verification")
    print("="*60)
    
    # Load original data
    df = preprocess.load_data(config.DATA_FILE, config)
    total_samples = len(df)
    
    print(f"\nTotal samples: {total_samples}")
    print(f"\nConfigured split:")
    print(f"  Test split:  {config.TEST_SPLIT*100}%")
    print(f"  Val split:   {config.VAL_SPLIT*100}%")
    
    # Split data
    train_df, val_df, test_df = preprocess.split_data(df, config)
    
    print(f"\nActual split:")
    print(f"  Train:       {len(train_df):5d} samples ({len(train_df)/total_samples*100:.1f}%)")
    print(f"  Validation:  {len(val_df):5d} samples ({len(val_df)/total_samples*100:.1f}%)")
    print(f"  Test:        {len(test_df):5d} samples ({len(test_df)/total_samples*100:.1f}%)")
    print(f"  Total:       {len(train_df)+len(val_df)+len(test_df):5d} samples")
    
    # Check for data leakage
    train_texts = set(train_df['text'].values)
    val_texts = set(val_df['text'].values)
    test_texts = set(test_df['text'].values)
    
    overlap_train_val = train_texts & val_texts
    overlap_train_test = train_texts & test_texts
    overlap_val_test = val_texts & test_texts
    
    print(f"\nData leakage check:")
    print(f"  Train ∩ Val:  {len(overlap_train_val)} samples")
    print(f"  Train ∩ Test: {len(overlap_train_test)} samples")
    print(f"  Val ∩ Test:   {len(overlap_val_test)} samples")
    
    if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
        print("\n✅ No data leakage - splits are clean!")
    else:
        print("\n⚠️  Warning: Data leakage detected!")
    
    # Show sample from each split
    print(f"\nSample from each split:")
    print("-"*60)
    
    print("\nTrain sample:")
    print(f"  Text:  {train_df.iloc[0]['text']}")
    print(f"  Gloss: {train_df.iloc[0]['gloss']}")
    
    print("\nValidation sample:")
    print(f"  Text:  {val_df.iloc[0]['text']}")
    print(f"  Gloss: {val_df.iloc[0]['gloss']}")
    
    print("\nTest sample:")
    print(f"  Text:  {test_df.iloc[0]['text']}")
    print(f"  Gloss: {test_df.iloc[0]['gloss']}")
    
    print("\n" + "="*60)
    print("✓ Data split verification completed!")


if __name__ == "__main__":
    main()

