#!/usr/bin/env python3
"""inspect_weights.py
python3 scripts/inspect_weights.py weights/*.pth
python3 scripts/inspect_weights.py --mmap weights/dinov3_vit7b16*.pth   # 27 GB, low RAM
python3 scripts/inspect_weights.py --forward weights/dinov3_vitl16*.pth # needs RAM for the model
"""
import argparse
import sys
from collections import Counter

try:
    import torch
except ImportError:
    sys.exit("torch not installed — `pip install torch` or enable torchCuda in flake.nix")


def load_any(path: str, use_mmap: bool):
    """Load a checkpoint and return the state_dict, descending into a wrapper if present."""
    kw = dict(map_location="cpu")
    if use_mmap:
        kw["mmap"] = True
    try:
        obj = torch.load(path, weights_only=True, **kw)
    except TypeError:  # torch < 2.0
        obj = torch.load(path, **kw)
    return obj


def describe(path: str, use_mmap: bool) -> dict:
    print("=" * 78)
    print(path)
    sd = load_any(path, use_mmap)

    nested = [k for k, v in sd.items() if isinstance(v, dict)] if isinstance(sd, dict) else []
    if nested:
        print(f"  wrapper keys at top level: {nested}")
        if len(nested) == 1:
            sd = sd[nested[0]]
            print(f"  -> descended into '{nested[0]}'")

    if not isinstance(sd, dict):
        print(f"  !! unexpected object type: {type(sd)}")
        return {}

    heads = [k for k in sd if k.startswith(("dino_head.", "ibot_head."))]
    has_heads = bool(heads)
    prefixes = Counter(k.split(".")[0] for k in sd)
    n_params = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    dtypes = Counter(str(v.dtype) for v in sd.values() if torch.is_tensor(v))

    print(f"  tensors: {len(sd)}    params: {n_params / 1e6:,.1f}M    dtypes: {dict(dtypes)}")
    print(f"  top-level prefixes: {dict(prefixes.most_common(8))}")
    print(f"  HAS dino_head/ibot_head: {has_heads}"
          f"{'  (' + str(len(heads)) + ' tensors)' if has_heads else ''}")

    if has_heads:
        print("  => usable directly as a distillation teacher (skip plan Step 3)")
    else:
        print("  => backbone only. The distillation loader will fail.")
        print("     PLAN.md Step 3 (adapt the teacher's heads) is required.")

    # Infer the architecture family from the keys — handy when names are opaque.
    keys = " ".join(list(sd)[:400])
    if "downsample_layers" in keys:
        print("  family: ConvNeXt")
    elif "blocks." in keys or "rope_embed" in keys:
        print("  family: ViT")
    return {"path": path, "has_heads": has_heads, "params": n_params}


def forward_check(path: str, repo: str) -> None:
    """Build the model and print its token grid — validates PLAN.md §4.3 by hand."""
    print("\n--- forward check (needs torch.hub + the repo clone) ---")
    for size in (256, 512):
        try:
            m = torch.hub.load(repo, "dinov3_vitl16", source="local", weights=path,
                               trust_repo=True) if "vitl16" in path else None
            if m is None:
                print("  (skipping: forward check only implemented for dinov3_vitl16)")
                return
            m.eval()
            with torch.no_grad():
                out = m.forward_features(torch.randn(1, 3, size, size))
            t = out["x_norm_patchtokens"].shape[1]
            print(f"  {size}px -> {t} patch tokens  (grid {int(t**0.5)}x{int(t**0.5)})")
        except Exception as e:  # noqa: BLE001 - we only want the message
            print(f"  {size}px failed: {type(e).__name__}: {e}")
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--mmap", action="store_true",
                    help="lazy load (torch>=2.0); use for the 27 GB ViT-7B on <=32 GB RAM")
    ap.add_argument("--forward", action="store_true", help="also build the model and run a forward pass")
    ap.add_argument("--repo", default="dinov3", help="path to the dinov3 clone (for --forward)")
    args = ap.parse_args()

    rows = [describe(p, args.mmap) for p in args.ckpts]

    print("\n" + "=" * 78)
    print("SUMMARY")
    for r in rows:
        if r:
            print(f"  {r['has_heads']!s:>5}  heads   {r['params'] / 1e6:>8,.1f}M   {r['path']}")

    if args.forward:
        for p in args.ckpts:
            forward_check(p, args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
