"""RemoteCLIP loader for the O-TPT backbone-swap pilot.

RemoteCLIP (Liu et al., 2023) is a CLIP variant fine-tuned for remote sensing.
The published checkpoint on HF Hub (`chendelong/RemoteCLIP`, file
`RemoteCLIP-ViT-B-32.pt`) uses an OpenAI-CLIP-compatible state_dict layout for
the ViT-B/32 architecture — all 302 tensors match the vendored `clip.load()`
model's keys and shapes exactly (verified during pilot bring-up). No key
translation is required; we simply build the vendored OpenAI CLIP ViT-B/32 and
overwrite its weights.

Public API:
    load_remoteclip(device='cpu', download_root=None) -> (model, preprocess)

`model` is exactly the object O-TPT's `PromptLearner` / `TextEncoder` expect
(`.visual`, `.transformer`, `.token_embedding`, `.positional_embedding`,
 `.ln_final`, `.text_projection`, `.logit_scale`, `.dtype`).
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OTPT_BASE = os.path.join(_REPO_ROOT, "otpt-base")
if _OTPT_BASE not in sys.path:
    sys.path.insert(0, _OTPT_BASE)

from clip import load as _clip_load  # vendored OpenAI CLIP


REMOTECLIP_REPO = "chendelong/RemoteCLIP"
REMOTECLIP_FILE = "RemoteCLIP-ViT-B-32.pt"
DEFAULT_CACHE = os.path.expanduser("~/.cache/remoteclip")
DEFAULT_CLIP_DOWNLOAD = os.path.expanduser("~/.cache/clip")


def _fetch_remoteclip_checkpoint(cache_dir: str) -> str:
    """Download the RemoteCLIP ViT-B-32 checkpoint via huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to fetch RemoteCLIP. "
            "pip install huggingface_hub"
        ) from e
    os.makedirs(cache_dir, exist_ok=True)
    return hf_hub_download(REMOTECLIP_REPO, REMOTECLIP_FILE, cache_dir=cache_dir)


def _normalize_state_dict(sd) -> dict:
    """Handle checkpoint wrappers: {'state_dict': ...} and 'module.' prefixes."""
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


def load_remoteclip(
    device: str = "cpu",
    download_root: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, object]:
    """Load RemoteCLIP ViT-B/32 into an OpenAI-CLIP-shaped model.

    Args:
        device: Torch device string. Defaults to "cpu" (transfer later).
        download_root: Where to keep the vendored CLIP arch checkpoint (the
            architecture template; separate from RemoteCLIP weights).
        checkpoint_path: Explicit RemoteCLIP weights path. If None, downloads
            from HF Hub into ~/.cache/remoteclip.

    Returns:
        (model, preprocess). `model` matches O-TPT's expected CLIP surface.
    """
    if download_root is None:
        download_root = DEFAULT_CLIP_DOWNLOAD

    model, _, preprocess = _clip_load(
        "ViT-B/32", device=device, download_root=download_root
    )

    if checkpoint_path is None:
        checkpoint_path = _fetch_remoteclip_checkpoint(DEFAULT_CACHE)

    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = _normalize_state_dict(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"RemoteCLIP state_dict does not match ViT-B/32 exactly.\n"
            f"  missing keys  ({len(missing)}): {list(missing)[:5]}\n"
            f"  unexpected   ({len(unexpected)}): {list(unexpected)[:5]}\n"
            "The published checkpoint layout may have changed — update this loader."
        )

    model.eval()
    return model, preprocess


if __name__ == "__main__":
    m, p = load_remoteclip(device="cpu")
    print("RemoteCLIP loaded.")
    print("  visual:", type(m.visual).__name__)
    print("  transformer:", type(m.transformer).__name__)
    print("  ctx_dim:", m.ln_final.weight.shape[0])
    print("  text_projection:", tuple(m.text_projection.shape))
    print("  logit_scale:", float(m.logit_scale.exp()))
