# O-TPT Backbone-Swap — Pilot Results

Domain-specific backbone pilots for O-TPT (CVPR'25). One small dataset per backbone to validate the end-to-end pipeline before scaling.

## Pilot table

| Dataset     | Backbone   | Zero-shot Acc | Post-TPT Acc | Post-TPT ECE | Notes |
|-------------|------------|---------------|--------------|--------------|-------|
| EuroSAT     | RemoteCLIP | _pending_     | _pending_    | _pending_    | To be filled from remote GPU run. Local CPU smoke-tested only. |
| DermaMNIST  | BioMedCLIP | _pending_     | _pending_    | _pending_    | To be filled from remote GPU run. Local CPU smoke-tested only. |

**Protocol** matches `otpt-base/scripts/test_tpt_otpt_fg.sh`:
- `--run_type tpt_otpt` (O-TPT with orthogonality regularizer)
- `--tta_steps 1`, `--lr 5e-3`, `--n_ctx 4`, `--ctx_init a_photo_of_a`, `--lambda_term 18`
- `-b 64` (63 AugMix views + 1 original), single test-time optimizer step per image (AdamW on prompt_learner.ctx)

## Smoke-test results (local, macOS/CPU, subset)

| Check | Backbone | Result |
|-------|----------|--------|
| Model loads + expected attrs present         | RemoteCLIP | _pending_ |
| Zero-shot top-1 on first 20 EuroSAT test    | RemoteCLIP | _pending_ |
| `ctx.grad` non-None after one TTA backward   | RemoteCLIP | _pending_ |
| Parity cosine-sim vs `open_clip.encode_text` | BioMedCLIP | _pending_ |
| Zero-shot top-1 on first 20 DermaMNIST-224   | BioMedCLIP | _pending_ |
| `ctx.grad` non-None after one TTA backward   | BioMedCLIP | _pending_ |

## Design decisions to flag to advisor

1. **EuroSAT label order.** Using torchvision's `EuroSAT` (alphabetical class order) rather than the CoOp/TPT JSON-split layout the upstream O-TPT ships with. `eurosat_tv_classes` in `data/cls_to_names.py` must match torchvision's `.classes` order exactly; label index i therefore maps to the same string upstream at i but sometimes to a **different** i in the fewshot split file. Small accuracy drift vs the paper's number is expected.
2. **RemoteCLIP checkpoint format.** RemoteCLIP publishes open_clip-formatted state dicts; O-TPT expects OpenAI-CLIP layout. We translate keys at load time in `backbones/remoteclip_loader.py`. If future RemoteCLIP releases change the layout the translator will need an update — flagged in code.
3. **BioMedCLIP text tower is BERT (PubMedBERT).** New `PromptLearnerBERT` / `TextEncoderBERT` / `ClipTestTimeTuningBERT` in `clip/custom_clip_biomedclip.py`. The BERT forward pass is replayed manually (embeddings LayerNorm → encoder → CLS pool → projection) so gradients flow to the learned `ctx` prompt tokens. `n_ctx=4` shared context, same as RemoteCLIP.
4. **DermaMNIST resolution.** Using the 224×224 variant (`medmnist>=3.0`) rather than the 28×28 default, so images actually resemble BioMedCLIP's pretraining distribution.
5. **Device-agnostic loop.** Patched `otpt_classification.py` to drop hard `.cuda(args.gpu)` and `torch.cuda.amp.autocast` calls; CUDA path is preserved bit-identically on GPU boxes but CPU/MPS boxes can run the same code (used for smoke tests here).
6. **Not run this session.** The pilot itself (full test set, per O-TPT paper protocol) is deferred to a remote GPU. This laptop cannot run 64 AugMix views × ~2,700 test images through ViT-B/16 in a reasonable time.

## Open follow-ups

- Confirm chosen prompt template `"a photo of a {classname}."` is the right init for domain-specific backbones (RemoteCLIP paper uses `"a satellite image of ..."`; BioMedCLIP prefers `"this is a photo of {}"`). Kept the upstream O-TPT init for pilot parity; worth an ablation later.
- Confirm orthogonality-loss `lambda_term=18` is a sane default when the text encoder is BERT-shaped rather than CLIP-shaped. Same 512-D output space, so mechanically fine, but conditioning is different.
- Full dataset suite scale-up: RemoteCLIP → RESISC45, AID, PatternNet, WHU-RS19, UC Merced, MillionAID; BioMedCLIP → PneumoniaMNIST, ChestX-ray, PathMNIST, RETINAMNIST, OrganAMNIST, PCAM (per advisor's dataset lists).
