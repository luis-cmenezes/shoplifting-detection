"""Count trainable parameters for every TS and I3D variant."""

import torch
import sys
sys.path.insert(0, "timesformer-shoplifting/src")
sys.path.insert(0, "i3d-shoplifting/src")

from timesformer_shoplifting.models.model_utils import get_model_and_processor, set_freeze_strategy
from i3d_shoplifting.models.i3d_pytorch import InceptionI3d

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen

print("=" * 70)
print("I3D")
print("=" * 70)

for mode in ["full", "head"]:
    model = InceptionI3d(num_classes=400, in_channels=3)
    model.replace_logits(1)
    if mode == "head":
        # Freeze backbone, only logits trainable
        for param in model.parameters():
            param.requires_grad = False
        for param in model.logits.parameters():
            param.requires_grad = True
    total, trainable, frozen = count_params(model)
    print(f"  I3D {mode:>5}: total={total:>12,}  trainable={trainable:>12,}  frozen={frozen:>12,}")

print()
print("=" * 70)
print("TimeSformer")
print("=" * 70)

for ckpt_name, ckpt_id in [("K400", "facebook/timesformer-base-finetuned-k400"),
                             ("SSv2", "facebook/timesformer-base-finetuned-ssv2")]:
    for nf in [8, 32, 64]:
        model, processor, is_interp = get_model_and_processor(
            model_name=ckpt_id,
            num_labels=2,
            num_frames=nf,
        )
        for strategy in ["unfreeze_all", "unfreeze_head"]:
            set_freeze_strategy(model, strategy, unfreeze_time_embeddings=is_interp)
            total, trainable, frozen = count_params(model)
            label = f"{ckpt_name}/{nf}f/{strategy}"
            print(f"  {label:<40s}: total={total:>12,}  trainable={trainable:>12,}  frozen={frozen:>12,}")
        print()
