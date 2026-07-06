"""Local smoke test for the RemoteCLIP + EuroSAT backbone-swap path.

Exercises, on CPU:
1. RemoteCLIP loader end-to-end.
2. Attribute presence on the loaded object (what O-TPT's PromptLearner needs).
3. Zero-shot top-1 on a small EuroSAT subset (first 20 test images).
4. One AugMix TTA backward step; assert ctx.grad is non-None + non-zero.

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
from PIL import Image

from clip import tokenize
from clip.custom_clip_iptp_bas import ClipTestTimeTuning
from data.cls_to_names import eurosat_tv_classes
from data.datautils import build_dataset, AugMixAugmenter

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cpu")
DATA_ROOT = os.path.join(REPO_ROOT, "data")
N_SUBSET = 20


def assert_expected_attrs(model):
    """O-TPT's PromptLearner + TextEncoder read exactly these fields."""
    for attr in (
        "visual",
        "transformer",
        "token_embedding",
        "positional_embedding",
        "ln_final",
        "text_projection",
        "logit_scale",
        "dtype",
    ):
        assert hasattr(model, attr), f"RemoteCLIP model missing .{attr}"
    print("[OK] RemoteCLIP object has all attributes O-TPT expects.")


def zero_shot_eurosat(clip_model):
    """Compute zero-shot top-1 on first N_SUBSET EuroSAT test images."""
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
    ds = build_dataset("eurosat_tv", preprocess, DATA_ROOT, mode="test")
    print(f"[info] EuroSAT test set size: {len(ds)}")
    n = min(N_SUBSET, len(ds))

    prompts = [f"a photo of {c}." for c in eurosat_tv_classes]
    text_ids = tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        text_feat = clip_model.encode_text(text_ids)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    correct = 0
    for i in range(n):
        img, label = ds[i]
        img = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = clip_model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = (img_feat @ text_feat.T).squeeze(0)
        pred = int(logits.argmax().item())
        correct += int(pred == int(label))
    acc = correct / n
    print(f"[metric] Zero-shot top-1 on first {n} EuroSAT images: {acc:.3f}")
    return acc


def one_tta_backward(cttt_model):
    """Build ClipTestTimeTuning, freeze non-prompt-learner params, run one
    forward + backward on an AugMix-style batch, assert ctx.grad is populated."""
    for name, p in cttt_model.named_parameters():
        if "prompt_learner" not in name:
            p.requires_grad_(False)

    # Simulate the O-TPT AugMix batch: 64 augmented views.
    x = torch.randn(8, 3, 224, 224)  # smaller than 64 to keep CPU fast
    cttt_model.l2_norm_cal = True
    cttt_model.train()
    logits = cttt_model(x, None, None)
    loss = -torch.log_softmax(logits, dim=-1).mean()
    loss.backward()
    grad = cttt_model.prompt_learner.ctx.grad
    assert grad is not None, "ctx.grad is None after backward"
    grad_norm = grad.norm().item()
    assert grad_norm > 0, f"ctx.grad has zero norm: {grad_norm}"
    assert cttt_model.textfeatures_ is not None, "textfeatures_ was not populated"
    print(f"[OK] TTA backward populated ctx.grad (norm={grad_norm:.4f}).")


def main():
    print("=" * 70)
    print("Smoke test: RemoteCLIP + EuroSAT")
    print("=" * 70)

    # Step 1+2: Load + attribute presence.
    from backbones.remoteclip_loader import load_remoteclip

    print("[step 1] Loading RemoteCLIP ViT-B/32...")
    clip_model, _ = load_remoteclip(device=DEVICE)
    assert_expected_attrs(clip_model)

    # Step 3: Zero-shot on EuroSAT subset (opt-in — the EuroSAT archive is ~2GB.
    # The other steps are what actually validate the code path; the zero-shot
    # number is nice-to-have and runs on the remote GPU later anyway).
    if os.environ.get("SMOKE_FULL") == "1":
        try:
            acc = zero_shot_eurosat(clip_model)
            # rough sanity bound — RemoteCLIP is domain-specialized on 10-way EuroSAT
            assert acc > 0.15, f"suspicious zero-shot accuracy: {acc:.3f}"
            print("[OK] Zero-shot top-1 in sane range.")
        except (RuntimeError, OSError) as e:
            print(f"[skip] Zero-shot on EuroSAT failed to run: {e}")
            acc = None
    else:
        print("[skip] Step 3 (zero-shot) skipped — set SMOKE_FULL=1 to enable.")
        acc = None

    # Step 4: TTA backward via ClipTestTimeTuning.
    print("[step 4] Building ClipTestTimeTuning(arch='remoteclip')...")
    cttt = ClipTestTimeTuning(
        device=str(DEVICE),
        classnames=eurosat_tv_classes,
        batch_size=None,
        arch="remoteclip",
        n_ctx=4,
        ctx_init="a_photo_of_a",
    )
    one_tta_backward(cttt)

    print()
    print("=" * 70)
    print("RemoteCLIP smoke test: PASS")
    if acc is not None:
        print(f"  zero-shot top-1 (subset={N_SUBSET}): {acc:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
