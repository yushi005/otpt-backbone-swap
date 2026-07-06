"""BioMedCLIP (PubMedBERT) path for the O-TPT backbone-swap pilot.

BioMedCLIP's text tower is a HuggingFace BertModel (PubMedBERT), not the causal
OpenAI-CLIP transformer. This module provides:

- PromptLearnerBERT: injects a learnable ctx tensor between [CLS] and the
  classname wordpieces, mirroring O-TPT's CLIP-side PromptLearner API.
- TextEncoderBERT: replays BERT internals (word_emb replaced by our fused
  embedding, then pos_emb + tt_emb + LayerNorm + dropout + encoder + CLS pool +
  proj) so gradients flow to ctx.
- ClipTestTimeTuningBERT: composition wrapper that exposes the public surface
  otpt_classification.py needs (prompt_learner, image_encoder, logit_scale,
  reset, reset_classnames, get_text_features, inference, forward, dtype,
  textfeatures_, l2_norm_cal).
- get_coop_biomedclip: factory mirroring custom_clip_iptp_bas.get_coop.

Verified against open_clip's HFTextEncoder.forward at BiomedCLIP bring-up: the
reference path is `transformer(input_ids=x, attention_mask=(x!=pad).long())`
then `last_hidden_state[:, 0]` then `proj(...)`. Our replay must match this
to within numerical noise (>0.999 cosine sim) when ctx equals the actual
word embeddings of the CLIP-init prompt.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *  # noqa: F401,F403 — dataset classnames lookup


BIOMEDCLIP_HF_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMEDCLIP_OPEN_CLIP_ID = f"hf-hub:{BIOMEDCLIP_HF_ID}"


def _load_biomedclip(device):
    """Load BiomedCLIP via open_clip. Returns (model, preprocess, tokenizer_hf).

    tokenizer_hf is the raw HF BertTokenizer, not the open_clip wrapper — we
    need `add_special_tokens=False` control that the wrapper does not expose.
    """
    import open_clip
    from transformers import AutoTokenizer

    model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP_OPEN_CLIP_ID)
    model = model.to(device)
    model.eval()
    tokenizer_hf = AutoTokenizer.from_pretrained(BIOMEDCLIP_HF_ID)
    return model, preprocess, tokenizer_hf


class PromptLearnerBERT(nn.Module):
    """BERT-side analog of PromptLearner.

    Layout of each per-class token sequence (fixed length L):
        [CLS] [ctx x n_ctx] [wordpieces of classname] [SEP] [PAD ...]

    - `ctx` is a single (n_ctx, hidden=768) Parameter shared across classes.
    - `token_prefix` buffer holds the [CLS] row for each class ([n_cls, 1, 768]).
    - `token_suffix` buffer holds [wordpieces + [SEP] + [PAD]] rows for each
      class ([n_cls, L-1-n_ctx, 768]). Padding is handled by the attention
      mask; gradient does not flow through pad positions.
    - Attention mask, token_type_ids, position_ids are stored as buffers.

    forward() returns the fused input-embedding tensor [n_cls, L, 768] that
    TextEncoderBERT then passes through BERT.
    """

    def __init__(
        self,
        biomedclip_model,
        tokenizer_hf,
        classnames,
        n_ctx: int = 4,
        ctx_init=None,
        slack: int = 2,
    ):
        super().__init__()
        self.tokenizer = tokenizer_hf
        self.bert = biomedclip_model.text.transformer
        self.hidden_size = self.bert.config.hidden_size
        self.device = next(self.bert.parameters()).device
        self.dtype = next(self.bert.parameters()).dtype
        self.slack = slack

        classnames = [c.replace("_", " ") for c in classnames]
        self.classnames = classnames

        # Resolve ctx_init (n_ctx and prompt_prefix string).
        if ctx_init:
            print(f"Initializing the contect with given words: [{ctx_init}]")
            prompt_prefix = ctx_init.replace("_", " ")
            n_ctx = len(prompt_prefix.split(" "))
        else:
            print("Random initialization: initializing a generic context")
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.n_ctx = n_ctx
        self.prompt_prefix = prompt_prefix
        self.ctx_init = ctx_init
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # Build classnames buffers + ctx parameter.
        self._build_class_buffers(classnames)

        # Initialize ctx from the actual PubMedBERT word embeddings of
        # prompt_prefix if ctx_init was provided (mirrors CLIP-side behavior).
        ctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        if ctx_init:
            with torch.no_grad():
                ids = self.tokenizer(
                    prompt_prefix, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(self.device)
                if ids.shape[1] >= n_ctx:
                    emb = self.bert.embeddings.word_embeddings(ids[0, :n_ctx])
                    ctx_vectors = emb.detach().clone().to(dtype=self.dtype)
                else:
                    print(
                        f"[warn] ctx_init '{prompt_prefix}' tokenizes to "
                        f"{ids.shape[1]} pieces, less than n_ctx={n_ctx}; "
                        "using random init."
                    )
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)

    def _build_class_buffers(self, classnames):
        """Tokenize each classname, build per-class attention_mask, position_ids,
        token_type_ids, and (word-embedded) prefix/suffix buffers."""
        tokenizer = self.tokenizer
        n_ctx = self.n_ctx

        # Wordpieces per class, no special tokens.
        wp_ids = [
            tokenizer(name, add_special_tokens=False)["input_ids"]
            for name in classnames
        ]
        name_lens = [len(w) for w in wp_ids]
        max_name_len = max(name_lens)
        # Total length = [CLS] + n_ctx + wordpieces + [SEP] + slack (pads).
        L = 1 + n_ctx + max_name_len + 1 + self.slack

        # Build [n_cls, L] input_ids with per-class layout.
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id
        # Placeholder for the ctx slots — anything valid; will be overwritten
        # by ctx before running BERT. Use pad_id so the sequence stays valid
        # even if some code path accidentally reads it.
        placeholder = pad_id

        n_cls = len(classnames)
        input_ids = torch.full((n_cls, L), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n_cls, L), dtype=torch.long)
        for i, wp in enumerate(wp_ids):
            input_ids[i, 0] = cls_id
            for j in range(n_ctx):
                input_ids[i, 1 + j] = placeholder
            for j, tid in enumerate(wp):
                input_ids[i, 1 + n_ctx + j] = tid
            input_ids[i, 1 + n_ctx + len(wp)] = sep_id
            # Attention over CLS + ctx + wordpieces + SEP.
            attention_mask[i, : 1 + n_ctx + len(wp) + 1] = 1

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(L, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            base_emb = self.bert.embeddings.word_embeddings(input_ids)  # [n_cls, L, hidden]

        # Buffers.
        self.register_buffer("token_prefix", base_emb[:, :1, :].clone())  # [n_cls, 1, H]
        self.register_buffer("token_suffix", base_emb[:, 1 + n_ctx :, :].clone())  # [n_cls, L-1-n_ctx, H]
        self.register_buffer("attention_mask", attention_mask)
        self.register_buffer("token_type_ids", token_type_ids)
        self.register_buffer("position_ids", position_ids)

        self.tokenized_prompts = input_ids  # exposed for parity with CLIP-side
        self.name_lens = name_lens
        self.n_cls = n_cls
        self.L = L

    def reset(self):
        with torch.no_grad():
            self.ctx.copy_(self.ctx_init_state)

    def reset_classnames(self, classnames, arch=None):
        """Rebuild per-class buffers for a new classname list. `arch` is
        accepted for API parity with the CLIP-side PromptLearner and ignored
        here — we reuse the already-loaded BERT and tokenizer."""
        classnames = [c.replace("_", " ") for c in classnames]
        self.classnames = classnames
        self._build_class_buffers(classnames)

    def forward(self):
        # Expand shared ctx to per-class shape.
        ctx = self.ctx  # [n_ctx, H]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, n_ctx, H]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, ctx, suffix], dim=1)  # [n_cls, L, H]
        return prompts


class TextEncoderBERT(nn.Module):
    """Manual replay of BiomedCLIP's HFTextEncoder that consumes fused input
    embeddings instead of input_ids, so gradients flow to the learned ctx.

    Matches open_clip's HFTextEncoder.forward:
        out = transformer(input_ids=x, attention_mask=am)
        pooled = out.last_hidden_state[:, 0]  # ClsLastHiddenStatePooler
        projected = proj(pooled)
    """

    def __init__(self, biomedclip_model):
        super().__init__()
        self.bert = biomedclip_model.text.transformer
        self.proj = biomedclip_model.text.proj  # Sequential(Linear, GELU, Linear)

    def forward(self, prompts_embeds, attention_mask, token_type_ids, position_ids):
        # BERT embeddings module already contains word_embeddings + position + tt + LN + dropout.
        # We must NOT call it on input_ids (we have already word-embedded and
        # spliced in ctx). Instead, add pos + tt manually and apply LN + dropout.
        embs = self.bert.embeddings
        pos_emb = embs.position_embeddings(position_ids)  # [1, L, H]
        tt_emb = embs.token_type_embeddings(token_type_ids)  # [n_cls, L, H]
        x = prompts_embeds + pos_emb + tt_emb
        x = embs.LayerNorm(x)
        x = embs.dropout(x)

        # Additive attention mask [n_cls, 1, 1, L] with 0/-inf.
        ext_mask = self.bert.get_extended_attention_mask(attention_mask, x.shape[:-1])

        enc = self.bert.encoder(x, attention_mask=ext_mask)
        last_hidden = enc.last_hidden_state  # [n_cls, L, H]

        pooled = last_hidden[:, 0]  # ClsLastHiddenStatePooler
        projected = self.proj(pooled)  # [n_cls, embed_dim]
        return projected


class ClipTestTimeTuningBERT(nn.Module):
    """BioMedCLIP-backed drop-in replacement for ClipTestTimeTuning.

    Exposes the public surface otpt_classification.py touches:
    - .image_encoder(image) returning L2-un-normalized image features.
    - .prompt_learner (has .ctx as its only Parameter).
    - .text_encoder (BERT replay).
    - .logit_scale (buffer).
    - .reset(), .reset_classnames(classnames, arch).
    - .get_text_features() — L2-normalized text features [n_cls, embed_dim].
    - .inference(image, cons, args) — logits [1, n_cls].
    - .forward(input, cons, args) — routes to .inference() for image tensors.
    - .textfeatures_ (set inside .inference when l2_norm_cal is True).
    - .l2_norm_cal (bool, toggled by the eval loop).
    - .dtype property.
    """

    def __init__(self, device, classnames, arch="biomedclip", n_ctx=4, ctx_init=None):
        super().__init__()
        model, preprocess, tokenizer_hf = _load_biomedclip(device)
        self.open_clip_model = model  # keep for parity checks (encode_text)
        self.preprocess = preprocess
        self.image_encoder = model.visual  # TimmModel; .forward(x) -> [B, 512]
        # logit_scale is a Parameter on open_clip model; freeze and store data
        # so the eval loop's `logit_scale.exp()` still works.
        self.register_buffer("logit_scale", model.logit_scale.data.detach().clone())

        self.prompt_learner = PromptLearnerBERT(
            model, tokenizer_hf, classnames, n_ctx=n_ctx, ctx_init=ctx_init
        )
        self.text_encoder = TextEncoderBERT(model)

        self.l2_norm_cal = False
        self.textfeatures_ = None

    @property
    def dtype(self):
        return next(self.image_encoder.parameters()).dtype

    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch=None):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        prompts = self.prompt_learner()
        text = self.text_encoder(
            prompts,
            self.prompt_learner.attention_mask,
            self.prompt_learner.token_type_ids,
            self.prompt_learner.position_ids,
        )
        text = text / text.norm(dim=-1, keepdim=True)
        return text

    def inference(self, image, cons, args):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.l2_norm_cal:
            self.textfeatures_ = text_features

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits

    def forward(self, input, cons, args):
        # Image tensor path (the only path exercised by the O-TPT eval loop
        # after we drop the cocoop/contrastive branches).
        return self.inference(input, cons, args)


def get_coop_biomedclip(clip_arch, test_set, device, n_ctx, ctx_init, cons, learned_cls=False):
    """Factory mirroring get_coop in custom_clip_iptp_bas.

    `cons`, `learned_cls` are accepted for API parity and ignored — the pilot
    uses neither the learnable-classname nor the disp-constraint branch.
    """
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == "eurosat_tv":
        from data.cls_to_names import eurosat_tv_classes
        classnames = eurosat_tv_classes
    elif test_set == "dermamnist":
        from data.cls_to_names import dermamnist_classes
        classnames = dermamnist_classes
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuningBERT(
        device, classnames, arch=clip_arch, n_ctx=n_ctx, ctx_init=ctx_init
    )
    return model
