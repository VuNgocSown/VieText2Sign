# Text2Gloss

Vietnamese text to sign language gloss translation module.

## Structure

```
text2gloss/
├── __init__.py          # Module initialization
├── train.py            # Training script
├── evaluation.py       # Evaluation utilities
├── preprocess.py       # Data preprocessing
├── metrics.py          # Metrics computation
├── model_utils.py      # Model utilities
├── logger_utils.py     # Logging utilities
├── config.py           # Configuration
├── data/              # Training data
│   └── Corpus-Vie-VSL-10K.xlsx
├── models/            # Saved models
└── logs/              # Training logs
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

Excel file with two columns:
- Column 1: Vietnamese text
- Column 2: Gloss notation

## Configuration

Edit `config.py`:
- `DATA_FILE`: Data file path
- `NUM_EPOCHS`: Training epochs (default: 80)
- `BATCH_SIZE`: Batch size (default: 32)
- `LEARNING_RATE`: Learning rate (default: 3e-5)

## Usage

```bash
cd /home/vnson/vnson/text2sign_pipeline
python -m text2gloss.train
```

## Evaluation

```bash
python -m text2gloss.evaluation
```

Metrics: BLEU-1/2/3/4, ROUGE-L, chrF, Token/Sequence Accuracy

## Prediction

```bash
python -m text2gloss.test_result
```

## Model

- Base: facebook/mbart-large-50
- Fine-tuning: Seq2Seq
- Metrics: BLEU-1/2/3/4, ROUGE-L, chrF, Token/Sequence Accuracy

## Benchmark

- BLEU-4: >50 (good), >70 (excellent)
- ROUGE-L: >60 (good), >80 (excellent)  
- Sequence Accuracy: >40% (good), >60% (excellent)
