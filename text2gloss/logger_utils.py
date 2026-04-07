"""Logging utilities for Text2Gloss training and evaluation"""
import logging
import os
from datetime import datetime


def setup_logger(name, log_dir='./text2gloss/logs', log_file=None):
    """
    Setup logger with both file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        log_file: Log filename (auto-generated if None)
    
    Returns:
        logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_{timestamp}.log'
    
    log_path = os.path.join(log_dir, log_file)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers
    
    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("="*80)
    logger.info(f"Logging initialized: {log_path}")
    logger.info("="*80)
    
    return logger, log_path


def log_metrics(logger, metrics, prefix=""):
    """Log metrics in a formatted way"""
    logger.info(f"\n{prefix} Metrics:")
    logger.info("-" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key:20s}: {value:.4f}")
        else:
            logger.info(f"  {key:20s}: {value}")
    logger.info("-" * 80)


def log_training_config(logger, config):
    """Log training configuration"""
    logger.info("\nTraining Configuration:")
    logger.info("-" * 80)
    logger.info(f"  Experiment:         {config.ACTIVE_EXPERIMENT}")
    logger.info(f"  Model:              {config.MODEL_CHECKPOINT}")
    logger.info(f"  Data file:          {config.DATA_FILE}")
    logger.info(f"  Epochs:             {config.NUM_EPOCHS}")
    logger.info(f"  Batch size:         {config.BATCH_SIZE}")
    eff = config.BATCH_SIZE * getattr(config, 'GRAD_ACCUM_STEPS', 1)
    logger.info(f"  Grad accum steps:   {getattr(config, 'GRAD_ACCUM_STEPS', 1)}  (effective batch: {eff})")
    logger.info(f"  Learning rate:      {config.LEARNING_RATE}")
    logger.info(f"  Warmup ratio:       {getattr(config, 'WARMUP_RATIO', 'N/A')}")
    logger.info(f"  Weight decay:       {getattr(config, 'WEIGHT_DECAY', 0.01)}")
    logger.info(f"  Label smoothing:    {config.LABEL_SMOOTHING}")
    logger.info(f"  Dropout rate:       {config.DROPOUT_RATE}")
    logger.info(f"  Max length:         {config.MAX_LENGTH}")
    logger.info(f"  Device:             {config.DEVICE}")
    _bf16 = getattr(config, 'BF16', False)
    if _bf16:
        logger.info(f"  Precision:          BF16 (FP16 disabled for mT5 stability)")
    elif config.FP16:
        logger.info(f"  Precision:          FP16")
    else:
        logger.info(f"  Precision:          FP32")
    logger.info(f"  Seed:               {config.SEED}")
    logger.info("-" * 80)


def log_dataset_info(logger, datasets):
    """Log dataset information"""
    logger.info("\nDataset Information:")
    logger.info("-" * 80)
    logger.info(f"  Train samples:      {len(datasets['train']):5d}")
    logger.info(f"  Validation samples: {len(datasets['validation']):5d}")
    logger.info(f"  Test samples:       {len(datasets['test']):5d}")
    logger.info(f"  Total samples:      {len(datasets['train']) + len(datasets['validation']) + len(datasets['test']):5d}")
    logger.info("-" * 80)


def log_evaluation_summary(logger, results, num_samples):
    """Log evaluation summary"""
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total samples evaluated: {num_samples}")
    logger.info("\nKey Metrics:")
    logger.info(f"  • BLEU-1:           {results.get('bleu1', 0):.2f}")
    logger.info(f"  • BLEU-2:           {results.get('bleu2', 0):.2f}")
    logger.info(f"  • BLEU-3:           {results.get('bleu3', 0):.2f}")
    logger.info(f"  • BLEU-4:           {results.get('bleu4', 0):.2f}")
    logger.info(f"  • ROUGE-L:          {results.get('rouge_l', 0):.2f}")
    logger.info(f"  • chrF:             {results.get('chrf', 0):.2f}")
    logger.info(f"  • Token Accuracy:   {results.get('token_acc', 0):.2f}%")
    logger.info(f"  • Exact Match:      {results.get('seq_acc', 0):.2f}%")
    logger.info("="*80)


def log_length_analysis(logger, length_results):
    """Log analysis by sentence length"""
    logger.info("\nAnalysis by Sentence Length:")
    logger.info("-" * 80)
    for bucket_name in ["Short (≤3)", "Medium (4-7)", "Long (≥8)"]:
        if bucket_name in length_results:
            data = length_results[bucket_name]
            logger.info(f"\n{bucket_name}:")
            logger.info(f"  Samples:     {data['count']:4d}")
            logger.info(f"  BLEU-4:      {data['bleu4']:.2f}")
            logger.info(f"  ROUGE-L:     {data['rouge_l']:.2f}")
            logger.info(f"  Exact Match: {data['seq_acc']:.2f}%")
    logger.info("-" * 80)


def log_sample_predictions(logger, predictions, references, max_samples=5):
    """Log sample predictions"""
    logger.info(f"\nSample Predictions (showing {min(max_samples, len(predictions))}):")
    logger.info("-" * 80)
    for i in range(min(max_samples, len(predictions))):
        match = "✓" if predictions[i] == references[i] else "✗"
        logger.info(f"\n{i+1}. {match}")
        logger.info(f"  Reference:  {references[i]}")
        logger.info(f"  Prediction: {predictions[i]}")
        if predictions[i] != references[i]:
            ref_tokens = references[i].split()
            pred_tokens = predictions[i].split()
            logger.info(f"  Tokens: {len(ref_tokens)} → {len(pred_tokens)}")
    logger.info("-" * 80)

