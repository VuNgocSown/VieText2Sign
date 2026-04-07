"""Model and tokenizer utilities - dùng Auto classes để hỗ trợ mọi Seq2Seq model"""
from typing import Any
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel
)


def get_tokenizer(config: Any) -> PreTrainedTokenizer:
    """
    Load tokenizer tương thích với mọi AutoSeq2Seq model.
    Tự động set src_lang / tgt_lang nếu model yêu cầu (mBART, NLLB).
    """
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)

    # Chỉ set src/tgt lang nếu model yêu cầu (mBART, NLLB) - mT5 không có
    if config.TOKENIZER_SRC_LANG and hasattr(tokenizer, 'src_lang'):
        tokenizer.src_lang = config.TOKENIZER_SRC_LANG
    if config.TOKENIZER_TGT_LANG and hasattr(tokenizer, 'tgt_lang'):
        tokenizer.tgt_lang = config.TOKENIZER_TGT_LANG

    return tokenizer


def get_model(config: Any) -> PreTrainedModel:
    """
    Load model tương thích với mọi AutoSeq2Seq checkpoint.
    Tự động apply dropout nếu model config hỗ trợ.
    """
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(config.MODEL_CHECKPOINT)

    # Apply dropout - chỉ set nếu model config có attribute đó
    if hasattr(model_config, 'dropout'):
        model_config.dropout = config.DROPOUT_RATE
    if hasattr(model_config, 'attention_dropout'):
        model_config.attention_dropout = config.DROPOUT_RATE
    if hasattr(model_config, 'dropout_rate'):  # mT5 dùng 'dropout_rate'
        model_config.dropout_rate = config.DROPOUT_RATE

    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.MODEL_CHECKPOINT,
        config=model_config
    )
    return model


def get_forced_bos_token_id(config: Any, tokenizer: PreTrainedTokenizer):
    """
    Lấy forced_bos_token_id phù hợp với model.
    - mBART / NLLB: cần set để chỉ định ngôn ngữ đích
    - mT5 / MarianMT: không dùng, trả về None
    """
    if not config.FORCED_BOS_LANG:
        return None

    # Thử lấy từ lang_code_to_id (mBART)
    if hasattr(tokenizer, 'lang_code_to_id'):
        lang_id = tokenizer.lang_code_to_id.get(config.FORCED_BOS_LANG)
        if lang_id is not None:
            return lang_id

    # Thử lấy từ convert_tokens_to_ids (NLLB)
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        lang_id = tokenizer.convert_tokens_to_ids(config.FORCED_BOS_LANG)
        if lang_id != tokenizer.unk_token_id:
            return lang_id

    return None
