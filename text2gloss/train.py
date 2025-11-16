"""Training script for Text2Gloss"""
from . import config
from . import model_utils
from . import preprocess
from . import metrics
from . import logger_utils
import numpy as np
import random
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_preds, tokenizer):
    """
    Compute comprehensive metrics for evaluation
    Metrics: BLEU-1/2/3/4, ROUGE-L, chrF, Token Accuracy, Sequence Accuracy
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Decode labels (replace padding token id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Format for computation
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Compute all metrics using our custom metrics module
    results = metrics.compute_all_metrics(
        references=decoded_labels,
        hypotheses=decoded_preds,
        level='word'
    )
    
    # Return metrics for tracking
    # Primary metric for model selection: BLEU-4
    return {
        "bleu1": results["bleu1"],
        "bleu2": results["bleu2"],
        "bleu3": results["bleu3"],
        "bleu4": results["bleu4"],
        "rouge_l": results["rouge_l"],
        "chrf": results["chrf"],
        "token_acc": results["token_acc"],
        "seq_acc": results["seq_acc"]
    }


def main():
    # Setup logger
    logger, log_path = logger_utils.setup_logger('train', log_dir='logs')
    
    logger.info("="*60)
    logger.info("Text2Gloss Training")
    logger.info("="*60)
    
    set_seed(config.SEED)
    logger.info(f"Random seed set to: {config.SEED}")
    
    # Log configuration
    logger_utils.log_training_config(logger, config)
    
    logger.info("\nLoading tokenizer...")
    tokenizer = model_utils.get_tokenizer(config)
    logger.info(f"Tokenizer loaded: {config.MODEL_CHECKPOINT}")
    
    logger.info("\nLoading data...")
    datasets = preprocess.prepare_datasets(config, tokenizer)
    logger_utils.log_dataset_info(logger, datasets)
    
    logger.info("\nLoading model...")
    model = model_utils.get_model(config).to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {config.MODEL_CHECKPOINT}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        label_smoothing_factor=config.LABEL_SMOOTHING,
        weight_decay=0.01,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        predict_with_generate=True,
        fp16=config.FP16,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="bleu4",
        greater_is_better=True,
        report_to="none",
        seed=config.SEED,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer)
    )
    
    logger.info("\nStarting training...")
    logger.info("="*60)
    
    import time
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    logger.info("="*60)
    logger.info(f"Training completed in {training_time/3600:.2f} hours ({training_time/60:.2f} minutes)")
    
    best_model_path = f"{config.MODEL_OUTPUT_DIR}/best_model"
    logger.info(f"\nSaving best model to {best_model_path}")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Model saved: {best_model_path}")
    logger.info(f"Total epochs: {config.NUM_EPOCHS}")
    logger.info(f"Training time: {training_time/3600:.2f} hours")
    logger.info("="*80)
    logger.info("\nTraining completed successfully! ✓")


if __name__ == "__main__":
    main()
