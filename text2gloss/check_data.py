"""Quick script to verify data loading"""
from . import config
from . import preprocess


def main():
    """Check if data loads correctly"""
    print("="*60)
    print("Data Check")
    print("="*60)
    
    try:
        # Load data
        df = preprocess.load_data(config.DATA_FILE, config)
        
        # Show statistics
        print(f"\nTotal samples: {len(df)}")
        print(f"\nFirst 5 samples:")
        print("-"*60)
        
        for idx, row in df.head(5).iterrows():
            print(f"\n{idx+1}.")
            print(f"Text:  {row['text']}")
            print(f"Gloss: {row['gloss']}")
        
        print("\n" + "="*60)
        print("✓ Data loaded successfully!")
        print("You can now run: python train.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease check:")
        print("1. File exists at:", config.DATA_FILE)
        print("2. File has 2 columns (text, gloss)")
        print("3. File is a valid Excel file (.xlsx)")


if __name__ == "__main__":
    main()

