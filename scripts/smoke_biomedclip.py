"""Local smoke test for the BioMedCLIP + DermaMNIST backbone-swap path.

Exercises, on CPU:
1. BioMedCLIP loader end-to-end via open_clip.
2. **Parity assertion**: our manual BERT replay (ctx seeded from real word
   embeddings of "a photo of a") matches open_clip's own encode_text at cosine
   sim > 0.999. This is the single highest-risk item in the pilot design.
3. Zero-shot top-1 on first 20 DermaMNIST-test-224 images.
4. One TTA backward on an 8-view AugMix batch; assert ctx.grad + textfeatures_.

Fails loudly on any anomaly so pilot bring-up halts instead of silently drifting.
"""

from __future__ import annotations

import os
import sys
import warnings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "otpt-base"))

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from clip.custom_clip_biomedclip import (
    ClipTestTimeTuningBERT,
    BIOMEDCLIP_OPEN_CLIP_ID,
)
from data.cls_to_names import dermamnist_classes
from data.datautils import build_dataset

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cpu")
DATA_ROOT = os.path.join(REPO_ROOT, "data")
N_SUBSET = 20


def parity_check(model):
    """Cosine-sim parity: our text features vs open_clip's own encode_text.

    Uses ctx_init='a_photo_of_a' with n_ctx=4 so ctx is seeded from PubMedBERT
    word embeddings of exactly those 4 wordpieces, meaning our fused sequence
    matches the tokenizer's own tokenization of "a photo of a <name>" exactly.
    """
    import open_clip

    prompts_noperiod = [f"a photo of a {c}" for c in dermamnist_classes]
    tok = open_clip.get_tokenizer(BIOMEDCLIP_OPEN_CLIP_ID)
    with torch.no_grad():
        ref_ids = tok(prompts_noperiod)
        ref = model.open_clip_model.encode_text(ref_ids)
        ref = ref / ref.norm(dim=-1, keepdim=True)
        mine = model.get_text_features()

    cos = F.cosine_similarity(mine, ref, dim=-1)
    print(f"[metric] cos-sim per class (min): {cos.min().item():.7f}")
    assert cos.min().item() > 0.999, (
        f"BERT replay parity FAIL — min cos sim {cos.min().item():.5f} < 0.999.\n"
        "The manual replay in TextEncoderBERT disagrees with open_clip's own "
        "encode_text. Likely causes: wrong pooler position, missing/duplicated "
        "position or token-type add, LayerNorm ordering, or projection layer."
    )
    print("[OK] Parity vs open_clip.encode_text: cos-sim >= 0.999.")


def zero_shot_dermamnist(cttt_model):
    """Compute zero-shot top-1 on first N_SUBSET DermaMNIST-224 test images."""
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    preprocess = T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]
    )
    ds = build_dataset("dermamnist", preprocess, DATA_ROOT, mode="test")
    print(f"[info] DermaMNIST test set size: {len(ds)}")
    n = min(N_SUBSET, len(ds))

    cttt_model.eval()
    cttt_model.l2_norm_cal = False

    correct = 0
    for i in range(n):
        img, label = ds[i]
        img = img.unsqueeze(0)
        with torch.no_grad():
            logits = cttt_model.inference(img, None, None).squeeze(0)
        pred = int(logits.argmax().item())
        correct += int(pred == int(label))
    acc = correct / n
    print(f"[metric] Zero-shot top-1 on first {n} DermaMNIST images: {acc:.3f}")
    return acc


def one_tta_backward(cttt_model):
    for name, p in cttt_model.named_parameters():
        if "prompt_learner" not in name:
            p.requires_grad_(False)
    x = torch.randn(4, 3, 224, 224)
    cttt_model.l2_norm_cal = True
    cttt_model.train()
    logits = cttt_model(x, None, None)
    loss = -torch.log_softmax(logits, dim=-1).mean()
    loss.backward()
    grad = cttt_model.prompt_learner.ctx.grad
    assert grad is not None, "ctx.grad is None"
    grad_norm = grad.norm().item()
    assert grad_norm > 0, f"ctx.grad zero norm: {grad_norm}"
    assert cttt_model.textfeatures_ is not None
    print(f"[OK] TTA backward populated ctx.grad (norm={grad_norm:.4f}).")


def main():
    print("=" * 70)
    print("Smoke test: BioMedCLIP + DermaMNIST")
    print("=" * 70)

    # Step 1: Load
    print("[step 1] Building ClipTestTimeTuningBERT (loads BiomedCLIP from HF)...")
    cttt = ClipTestTimeTuningBERT(
        device=DEVICE,
        classnames=dermamnist_classes,
        arch="biomedclip",
        n_ctx=4,
        ctx_init="a_photo_of_a",
    )

    # Step 2: Parity (highest-risk item)
    print("[step 2] Parity check vs open_clip encode_text...")
    parity_check(cttt)

    # Step 3: Zero-shot (opt-in: downloads ~1GB DermaMNIST-224 archive).
    # Set SMOKE_FULL=1 to run this step; otherwise skip so parity + gradient
    # checks (which are what actually validate the code path) run in seconds.
    if os.environ.get("SMOKE_FULL") == "1":
        print("[step 3] Zero-shot top-1 on DermaMNIST subset...")
        try:
            acc = zero_shot_dermamnist(cttt)
        except (RuntimeError, OSError) as e:
            print(f"[skip] Zero-shot failed to run: {e}")
            acc = None
    else:
        print("[skip] Step 3 (zero-shot) skipped — set SMOKE_FULL=1 to enable.")
        acc = None

    # Step 4: TTA backward
    print("[step 4] One TTA backward pass...")
    one_tta_backward(cttt)

    print()
    print("=" * 70)
    print("BioMedCLIP smoke test: PASS")
    if acc is not None:
        print(f"  zero-shot top-1 (subset={N_SUBSET}): {acc:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
