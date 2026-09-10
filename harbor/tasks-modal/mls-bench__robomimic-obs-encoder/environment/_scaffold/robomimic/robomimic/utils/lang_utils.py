import os

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"

LANG_EMB_OBS_KEY = "lang_emb"
_TOKENIZER = "openai/clip-vit-large-patch14"
_FALLBACK_DIM = 768
_MODEL = None
_TOKENIZER_OBJ = None
_LOAD_ATTEMPTED = False


def _load_clip_from_cache():
    global _MODEL, _TOKENIZER_OBJ, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _MODEL, _TOKENIZER_OBJ
    _LOAD_ATTEMPTED = True
    try:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        cache_dir = os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip"))
        _MODEL = CLIPTextModelWithProjection.from_pretrained(
            _TOKENIZER,
            cache_dir=cache_dir,
            local_files_only=True,
        ).eval()
        _TOKENIZER_OBJ = AutoTokenizer.from_pretrained(
            _TOKENIZER,
            cache_dir=cache_dir,
            local_files_only=True,
            TOKENIZERS_PARALLELISM=True,
        )
    except Exception:
        _MODEL = None
        _TOKENIZER_OBJ = None
    return _MODEL, _TOKENIZER_OBJ


def get_lang_emb(lang):
    if lang is None:
        return None

    model, tokenizer = _load_clip_from_cache()
    if model is None or tokenizer is None:
        return torch.zeros(_FALLBACK_DIM)

    tokens = tokenizer(
        text=lang,
        add_special_tokens=True,
        max_length=25,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    return model(**tokens)["text_embeds"].detach()[0]


def get_lang_emb_shape():
    return list(get_lang_emb("dummy").shape)
