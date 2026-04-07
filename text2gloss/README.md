# Text2Gloss

Vietnamese text to sign language gloss translation module using mBART.

## Project Structure

```
text2gloss/
├── __init__.py          # Module initialization
├── config.py           # Configuration parameters
├── train.py            # Main training script
├── evaluation.py       # Model evaluation utilities
├── preprocess.py       # Data preprocessing pipeline
├── metrics.py          # Evaluation metrics computation
├── model_utils.py      # Model and tokenizer utilities
├── logger_utils.py     # Logging utilities
├── test_result.py      # Prediction/testing script
├── data/               # Training data directory
│   └── Corpus-Vie-VSL-10K.xlsx
├── models/             # Saved model checkpoints
├── logs/                # Training logs
└── scripts/             # Analysis and testing scripts (optional)
    ├── README.md
    ├── analyze_characters.py
    ├── analyze_punctuation.py
    ├── check_end_punctuation.py
    ├── deep_analysis.py
    ├── test_preprocessing.py
    └── ...
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

Excel file with two columns (no header):
- Column 0: Vietnamese text
- Column 1: Gloss notation (sign language representation)

## Configuration

Edit `config.py` to adjust:
- `DATA_FILE`: Path to training data
- `NUM_EPOCHS`: Number of training epochs (default: 20)
- `BATCH_SIZE`: Batch size (default: 32)
- `LEARNING_RATE`: Learning rate (default: 3e-5)
- `MAX_LENGTH`: Maximum sequence length (default: 128)
- `TEST_SPLIT`, `VAL_SPLIT`: Data split ratios (default: 0.1 each)

## Usage

### Training

```bash
cd /home/vnson/vnson/text2sign_pipeline
python -m text2gloss.train
```

### Evaluation

```bash
python -m text2gloss.evaluation
```

### Prediction/Testing

```bash
python -m text2gloss.test_result
```

## Data Preprocessing

The preprocessing pipeline automatically applies:
1. BOM (Byte Order Mark) removal
2. Whitespace normalization
3. Unicode normalization (NFC)
4. Lowercase conversion
5. End punctuation normalization
6. Duplicate removal

## Metrics

The model is evaluated using:
- **BLEU-1/2/3/4**: N-gram precision scores
- **ROUGE-L**: Longest common subsequence
- **chrF**: Character-level F-score
- **Token Accuracy**: Token-level exact match
- **Sequence Accuracy**: Full sequence exact match

## Model

- **Base Model**: `facebook/mbart-large-50`
- **Architecture**: Sequence-to-Sequence (Seq2Seq)
- **Fine-tuning**: Full fine-tuning with label smoothing

## Performance Benchmarks

- **BLEU-4**: >50 (good), >70 (excellent)
- **ROUGE-L**: >60 (good), >80 (excellent)
- **Sequence Accuracy**: >40% (good), >60% (excellent)

## Scripts Directory

The `scripts/` directory contains optional analysis and testing scripts:
- Character analysis
- Punctuation analysis
- Dataset deep analysis
- Preprocessing tests

These scripts are not required for training and can be safely deleted if not needed.
See `scripts/README.md` for more details.

## License

[Add your license information here]
