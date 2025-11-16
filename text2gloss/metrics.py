"""Evaluation metrics for Text2Gloss translation task"""
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer


def chrf(references, hypotheses):
    """Character F-score from sacrebleu (0-100)"""
    score = sacrebleu.corpus_chrf(hypotheses=hypotheses, references=[references])
    return score.score


def bleu(references, hypotheses, level='word'):
    """Compute BLEU-1/2/3/4 scores with exponential smoothing"""
    if level == 'char':
        references = [' '.join(list(r)) for r in references]
        hypotheses = [' '.join(list(h)) for h in hypotheses]
    
    refs_formatted = [[ref] for ref in references]
    bleu_result = sacrebleu.corpus_bleu(
        hypotheses=hypotheses,
        references=list(zip(*refs_formatted)),
        smooth_method='exp',
        smooth_value=0.1,
        force=True,
        lowercase=False,
        use_effective_order=False
    )
    
    scores = {}
    for n in range(min(4, len(bleu_result.precisions))):
        scores[f'bleu{n + 1}'] = bleu_result.precisions[n]
    
    return scores


def rouge(references, hypotheses, level='word'):
    """Compute ROUGE-L F-score (0-100)"""
    if level == 'char':
        hypotheses = [' '.join(list(h)) for h in hypotheses]
        references = [' '.join(list(r)) for r in references]
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = [scorer.score(ref, hyp)['rougeL'].fmeasure 
              for hyp, ref in zip(hypotheses, references)]
    
    return np.mean(scores) * 100


def token_accuracy(references, hypotheses, level='word'):
    """Position-based token matching accuracy (0-100)"""
    correct_tokens = 0
    all_tokens = 0
    split_char = ' ' if level in ['word', 'bpe'] else ''
    
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.split(split_char) if split_char else list(hyp)
        ref_tokens = ref.split(split_char) if split_char else list(ref)
        all_tokens += len(hyp_tokens)
        correct_tokens += sum(1 for h, r in zip(hyp_tokens, ref_tokens) if h == r)
    
    return (correct_tokens / all_tokens) * 100 if all_tokens > 0 else 0.0


def sequence_accuracy(references, hypotheses):
    """Exact match accuracy (0-100)"""
    correct = sum(1 for hyp, ref in zip(hypotheses, references) if hyp == ref)
    return (correct / len(hypotheses)) * 100 if hypotheses else 0.0


def compute_all_metrics(references, hypotheses, level='word'):
    """Compute all evaluation metrics"""
    results = bleu(references, hypotheses, level=level)
    results['rouge_l'] = rouge(references, hypotheses, level=level)
    results['chrf'] = chrf(references, hypotheses)
    results['token_acc'] = token_accuracy(references, hypotheses, level=level)
    results['seq_acc'] = sequence_accuracy(references, hypotheses)
    return results


def format_scores(scores):
    """Format scores for logging"""
    lines = ["=" * 80, "EVALUATION RESULTS", "=" * 80]
    
    if any(k.startswith('bleu') for k in scores):
        lines.append("\nBLEU Scores:")
        for i in range(1, 5):
            if f'bleu{i}' in scores:
                lines.append(f"  BLEU-{i}: {scores[f'bleu{i}']:.2f}")
    
    if 'rouge_l' in scores:
        lines.append(f"\nROUGE-L: {scores['rouge_l']:.2f}")
    if 'chrf' in scores:
        lines.append(f"chrF:    {scores['chrf']:.2f}")
    
    if 'token_acc' in scores:
        lines.append(f"\nToken Accuracy:    {scores['token_acc']:.2f}%")
    if 'seq_acc' in scores:
        lines.append(f"Sequence Accuracy: {scores['seq_acc']:.2f}%")
    
    lines.append("=" * 80)
    return '\n'.join(lines)

