"""Model utilities for Text2Gloss"""
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBartConfig


def get_tokenizer(config):
    """Initialize tokenizer"""
    tokenizer = MBart50TokenizerFast.from_pretrained(config.MODEL_CHECKPOINT)
    tokenizer.src_lang = config.SOURCE_LANG
    tokenizer.tgt_lang = config.TARGET_LANG
    return tokenizer


def get_model(config):
    """Initialize model with dropout configuration"""
    model_config = MBartConfig.from_pretrained(config.MODEL_CHECKPOINT)
    model_config.dropout = config.DROPOUT_RATE
    model_config.attention_dropout = config.DROPOUT_RATE
    
    model = MBartForConditionalGeneration.from_pretrained(
        config.MODEL_CHECKPOINT,
        config=model_config
    )
    
    return model
