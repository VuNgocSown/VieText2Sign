"""Comprehensive evaluation script for Text2Gloss model"""
from . import config
from . import model_utils
from . import preprocess
from . import metrics
from . import logger_utils
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from tqdm import tqdm
from collections import defaultdict
import json
import os
import time


def evaluate_by_length(predictions, references):
    """Analyze performance by sentence length buckets"""
    length_buckets = defaultdict(lambda: {'predictions': [], 'references': []})
    
    for pred, ref in zip(predictions, references):
        ref_len = len(ref.split())
        
        if ref_len <= 3:
            bucket = "Short (≤3)"
        elif ref_len <= 7:
            bucket = "Medium (4-7)"
        else:
            bucket = "Long (≥8)"
        
        length_buckets[bucket]['predictions'].append(pred)
        length_buckets[bucket]['references'].append(ref)
    
    results = {}
    for bucket_name, bucket_data in length_buckets.items():
        if len(bucket_data['predictions']) > 0:
            bucket_scores = metrics.compute_all_metrics(
                references=bucket_data['references'],
                hypotheses=bucket_data['predictions'],
                level='word'
            )
            results[bucket_name] = {
                'count': len(bucket_data['predictions']),
                'bleu4': bucket_scores['bleu4'],
                'rouge_l': bucket_scores['rouge_l'],
                'seq_acc': bucket_scores['seq_acc']
            }
    
    return results


def evaluate_model(level='word'):
    """Comprehensive evaluation with multiple metrics"""
    # Setup logger
    logger, log_path = logger_utils.setup_logger('evaluation', log_dir='logs')
    
    logger.info("="*80)
    logger.info("Text2Gloss - Comprehensive Evaluation")
    logger.info("="*80)
    
    best_model_path = f"{config.MODEL_OUTPUT_DIR}/best_model"
    
    logger.info(f"\nLoading model from {best_model_path}")
    try:
        tokenizer = model_utils.get_tokenizer(config)
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path).to(config.DEVICE)
        model.eval()
        logger.info(f"Model loaded successfully on {config.DEVICE}")
    except Exception as e:
        logger.error(f"Error: {e}\nPlease run train.py first.")
        return
    
    logger.info("\nLoading test data...")
    datasets = preprocess.prepare_datasets(config, tokenizer)
    test_dataset = datasets["test"]
    logger.info(f"Test set size: {len(test_dataset)} samples")
    logger.info(f"Evaluation level: {level}")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        collate_fn=data_collator
    )
    
    all_predictions = []
    all_references = []
    
    logger.info("\nGenerating predictions...")
    start_time = time.time()
    
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=config.MAX_LENGTH,
                num_beams=4,
            )
        
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        labels_np = np.where(labels.cpu() != -100, labels.cpu(), tokenizer.pad_token_id)
        refs = tokenizer.batch_decode(labels_np, skip_special_tokens=True)
        
        all_predictions.extend([p.strip() for p in preds])
        all_references.extend([r.strip() for r in refs])
    
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f}s ({inference_time/len(all_predictions)*1000:.2f}ms per sample)")
    
    logger.info("\n" + "="*80)
    logger.info("COMPUTING METRICS")
    logger.info("="*80)
    
    results = metrics.compute_all_metrics(
        references=all_references,
        hypotheses=all_predictions,
        level=level
    )
    
    # Log results
    logger_utils.log_evaluation_summary(logger, results, len(all_predictions))
    
    length_results = evaluate_by_length(all_predictions, all_references)
    logger_utils.log_length_analysis(logger, length_results)
    
    logger_utils.log_sample_predictions(logger, all_predictions, all_references, max_samples=10)
    
    output_dir = config.MODEL_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    save_results = {
        'overall': results,
        'by_length': length_results,
        'num_samples': len(all_predictions),
        'level': level,
        'inference_time': inference_time,
        'avg_time_per_sample': inference_time / len(all_predictions),
        'config': {
            'model': config.MODEL_CHECKPOINT,
            'max_length': config.MAX_LENGTH,
            'batch_size': config.BATCH_SIZE,
            'device': str(config.DEVICE)
        }
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n💾 Results saved to: {results_file}")
    
    predictions_file = os.path.join(output_dir, 'predictions.txt')
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for ref, pred in zip(all_references, all_predictions):
            f.write(f"REF: {ref}\nHYP: {pred}\nMATCH: {ref == pred}\n{'-'*80}\n")
    logger.info(f"💾 Predictions saved to: {predictions_file}")
    
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Results file: {results_file}")
    logger.info(f"Predictions file: {predictions_file}")
    logger.info(f"Total samples: {len(all_predictions)}")
    logger.info(f"Inference time: {inference_time:.2f}s")
    logger.info(f"Avg time/sample: {inference_time/len(all_predictions)*1000:.2f}ms")
    logger.info("="*80)
    logger.info("\nEvaluation completed successfully! ✓")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Text2Gloss model')
    parser.add_argument('--level', type=str, default='word', 
                        choices=['word', 'char'],
                        help='Evaluation level: word or char')
    
    args = parser.parse_args()
    
    evaluate_model(level=args.level)
