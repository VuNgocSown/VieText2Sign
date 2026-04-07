import time
import numpy as np
import random
import torch
from typing import Dict
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    PreTrainedTokenizer,
    TrainerCallback
)

from . import config
from . import model_utils
from . import preprocess
from . import metrics
from . import logger_utils


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_preds: tuple, tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Clip predictions to valid token ID range [0, vocab_size-1]
    # Cần thiết với mT5/ViT5: có thể sinh token ID âm hoặc vượt range → OverflowError
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    results = metrics.compute_all_metrics(
        references=decoded_labels,
        hypotheses=decoded_preds,
        level='word'
    )
    
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


class MetricsLoggingCallback(TrainerCallback):
    """Callback to log evaluation metrics and track best results"""
    
    def __init__(self, logger):
        self.logger = logger
        self.best_metrics = {}
        self.best_epoch = 0
        self.best_bleu4 = -1.0
        self.all_epoch_metrics = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called after each logging step (including after eval metrics are computed).
        on_evaluate() is called BEFORE metrics are computed, so we use on_log()
        which is called AFTER Trainer appends eval_* keys to logs.
        """
        if logs is None:
            return
        
        # Chỉ xử lý khi là eval log (có eval_bleu4)
        if 'eval_bleu4' not in logs:
            return
        
        # Get epoch number
        if hasattr(state, 'epoch') and state.epoch is not None:
            epoch = int(state.epoch) if isinstance(state.epoch, (int, float)) else state.global_step
        else:
            epoch = getattr(state, 'global_step', 0)
        
        # Extract metrics
        eval_metrics = {
            'epoch': epoch,
            'bleu1': logs.get('eval_bleu1', 0.0),
            'bleu2': logs.get('eval_bleu2', 0.0),
            'bleu3': logs.get('eval_bleu3', 0.0),
            'bleu4': logs.get('eval_bleu4', 0.0),
            'rouge_l': logs.get('eval_rouge_l', 0.0),
            'chrf': logs.get('eval_chrf', 0.0),
            'token_acc': logs.get('eval_token_acc', 0.0),
            'seq_acc': logs.get('eval_seq_acc', 0.0),
        }
        
        self.all_epoch_metrics.append(eval_metrics)
        
        # Log current epoch metrics
        self.logger.info("\n" + "="*80)
        self.logger.info(f"EVALUATION - Epoch {epoch}")
        self.logger.info("="*80)
        logger_utils.log_metrics(self.logger, eval_metrics, prefix=f"Epoch {epoch}")
        
        # Track best metrics (based on bleu4)
        current_bleu4 = eval_metrics['bleu4']
        if current_bleu4 > self.best_bleu4:
            self.best_bleu4 = current_bleu4
            self.best_epoch = epoch
            self.best_metrics = eval_metrics.copy()
            self.logger.info(f"\n✓ New best model! BLEU-4: {current_bleu4:.4f} (Epoch {epoch})")
        
        self.logger.info("="*80 + "\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        if self.best_metrics:
            self.logger.info("\n" + "="*80)
            self.logger.info("BEST RESULTS DURING TRAINING")
            self.logger.info("="*80)
            self.logger.info(f"Best Epoch: {self.best_epoch}")
            logger_utils.log_metrics(self.logger, self.best_metrics, prefix="Best")
            self.logger.info("="*80)


def main() -> None:
    logger, log_path = logger_utils.setup_logger('train', log_dir='logs')
    
    logger.info("="*60)
    logger.info(f"Text2Gloss Training - Experiment: [{config.ACTIVE_EXPERIMENT}]")
    logger.info(f"Model: {config.MODEL_CHECKPOINT}")
    logger.info("="*60)
    
    set_seed(config.SEED)
    logger.info(f"Random seed: {config.SEED}")
    logger_utils.log_training_config(logger, config)
    
    logger.info("\nLoading tokenizer...")
    tokenizer = model_utils.get_tokenizer(config)
    logger.info(f"Tokenizer: {config.MODEL_CHECKPOINT}")
    
    logger.info("\nLoading data...")
    datasets = preprocess.prepare_datasets(config, tokenizer)
    logger_utils.log_dataset_info(logger, datasets)
    
    logger.info("\nLoading model...")
    model = model_utils.get_model(config).to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config.MODEL_CHECKPOINT}")
    logger.info(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Ấn định ngôn ngữ đích (chỉ áp dụng với mBART/NLLB, None với mT5/MarianMT)
    forced_bos_token_id = model_utils.get_forced_bos_token_id(config, tokenizer)
    model.config.forced_bos_token_id = forced_bos_token_id
    if forced_bos_token_id is not None:
        logger.info(f"forced_bos_token_id: {forced_bos_token_id} ({config.FORCED_BOS_LANG})")
    else:
        logger.info("forced_bos_token_id: None (not required for this model)")

    # Fix T5-style models (mT5, ViT5): decoder_start_token_id phải là pad_token_id
    # Nếu không set, mT5 sẽ generate sentinel tokens <extra_id_0> thay vì văn bản bình thường
    if getattr(model.config, 'model_type', '') in ['mt5', 't5']:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
        logger.info(f"[T5-fix] decoder_start_token_id = {tokenizer.pad_token_id} (pad_token_id)")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    effective_batch = config.BATCH_SIZE * config.GRAD_ACCUM_STEPS
    logger.info(f"Effective batch size: {effective_batch} ({config.BATCH_SIZE} × {config.GRAD_ACCUM_STEPS} accum steps)")
    logger.info(f"Warmup ratio: {config.WARMUP_RATIO}")
    
    # Log precision mode để dễ debug
    _bf16 = getattr(config, 'BF16', False)
    if _bf16:
        logger.info("Precision: BF16 (mT5-safe mixed precision)")
    elif config.FP16:
        logger.info("Precision: FP16")
    else:
        logger.info("Precision: FP32 (no mixed precision)")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        warmup_ratio=config.WARMUP_RATIO,        # Tăng LR dần trong giai đoạn đầu
        label_smoothing_factor=config.LABEL_SMOOTHING,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=1.0,                       # Gradient clipping chuẩn
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        predict_with_generate=True,
        fp16=config.FP16,
        bf16=_bf16,                              # BF16 thay thế FP16 cho mT5/ViT5
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="bleu4",
        greater_is_better=True,
        report_to="none",
        seed=config.SEED,
    )
    
    # Create callback for logging metrics
    metrics_callback = MetricsLoggingCallback(logger)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=[metrics_callback]
    )
    
    logger.info("\nStarting training...")
    logger.info("="*60)
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    logger.info("="*60)
    logger.info(f"Training completed in {training_time/3600:.2f} hours")
    
    best_model_path = f"{config.MODEL_OUTPUT_DIR}/best_model"
    logger.info(f"\nSaving model to {best_model_path}")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Log: {log_path}")
    logger.info(f"Model: {best_model_path}")
    logger.info(f"Epochs: {config.NUM_EPOCHS}")
    logger.info(f"Time: {training_time/3600:.2f} hours")
    
    # Log best results summary
    if metrics_callback.best_metrics:
        logger.info("\nBest Results Summary:")
        logger.info(f"  Best Epoch: {metrics_callback.best_epoch}")
        logger.info(f"  Best BLEU-4: {metrics_callback.best_metrics.get('bleu4', 0):.4f}")
        logger.info(f"  Best ROUGE-L: {metrics_callback.best_metrics.get('rouge_l', 0):.4f}")
        logger.info(f"  Best chrF: {metrics_callback.best_metrics.get('chrf', 0):.4f}")
        logger.info(f"  Best Token Acc: {metrics_callback.best_metrics.get('token_acc', 0):.4f}%")
        logger.info(f"  Best Exact Match: {metrics_callback.best_metrics.get('seq_acc', 0):.4f}%")
    
    logger.info("="*80)
    logger.info("\nTraining completed successfully! ✓")


if __name__ == "__main__":
    main()
