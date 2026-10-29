"""Cut Chesapeake Land Cover tiles and LoveDA images into crops (<out>/images/*.jpg, <out>/labels/*.png).

python -m satdistill.prepare chesapeake --splits pa-train pa-val pa-test --out data/crops/distill
python -m satdistill.prepare chesapeake --splits md-test --labels --out data/crops/ches_md_test
python -m satdistill.prepare loveda --split val --labels --out data/crops/loveda_val
"""
import argparse
import glob
import os
import random

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

from satdistill.utils import save_json

CHESAPEAKE_CLASSES = ["water", "tree canopy", "low vegetation", "barren", "impervious", "road"]
LOVEDA_CLASSES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
IGNORE = 255


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="dataset", required=True)
    ches = sub.add_parser("chesapeake")
    ches.add_argument("--root", default="data/chesapeake")
    ches.add_argument("--splits", nargs="+", required=True, help="state-split, e.g. pa-train md-test")
    ches.add_argument("--max-per-tile", type=int, default=200)
    love = sub.add_parser("loveda")
    love.add_argument("--root", default="data/loveda")
    love.add_argument("--split", choices=["train", "val"], required=True)
    love.add_argument("--scale", type=float, default=0.5, help="resize factor (0.5: 0.3 m to 0.6 m)")
    for p in (ches, love):
        p.add_argument("--out", required=True)
        p.add_argument("--labels", action="store_true", help="also save the label crops")
        p.add_argument("--size", type=int, default=256)
        p.add_argument("--max-nodata", type=float, default=0.05, help="maximum fraction of missing pixels")
        p.add_argument("--max-crops", type=int, default=0, help="0: no limit")
        p.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def grid(height, width, size):
    return [(r, c) for r in range(0, height - size + 1, size) for c in range(0, width - size + 1, size)]


def chesapeake_crops(args, rng):
    """Random non-overlapping windows of each tile, up to --max-per-tile."""
    tiles = []
    for split in args.splits:
        state, part = split.split("-")
        folder = "%s_1m_*_extended-debuffered-%s_tiles" % (state, part)
        found = sorted(glob.glob(os.path.join(args.root, "**", folder, "*_naip-new.tif"), recursive=True))
        print("%s: %d tiles" % (split, len(found)))
        tiles += found
    if not tiles:
        raise FileNotFoundError("no tiles for %s under %s" % (args.splits, args.root))
    rng.shuffle(tiles)
    for t, path in enumerate(tiles):
        print("tile %d / %d" % (t + 1, len(tiles)))
        tile = os.path.basename(path)[:-len("_naip-new.tif")]
        with rasterio.open(path) as naip, rasterio.open(path.replace("_naip-new.tif", "_lc.tif")) as lc:
            windows = grid(naip.height, naip.width, args.size)
            rng.shuffle(windows)
            for r, c in windows[:args.max_per_tile]:
                window = Window(c, r, args.size, args.size)
                image = naip.read([1, 2, 3], window=window).transpose(1, 2, 0)
                label = lc.read(1, window=window)
                label = np.where((label >= 1) & (label <= 6), label - 1, IGNORE).astype(np.uint8)
                yield "%s_%d_%d" % (tile, r, c), image, label


def loveda_crops(args, rng):
    """Non-overlapping windows of each resized LoveDA image."""
    pattern = os.path.join(args.root, "**", args.split.capitalize(), "*", "images_png", "*.png")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError("no LoveDA %s images under %s" % (args.split, args.root))
    rng.shuffle(paths)
    for path in paths:
        image = Image.open(path).convert("RGB")
        size = (round(image.width * args.scale), round(image.height * args.scale))
        image = np.asarray(image.resize(size, Image.BILINEAR))
        mask = np.asarray(Image.open(path.replace("images_png", "masks_png")).resize(size, Image.NEAREST))
        mask = np.where(mask == 0, IGNORE, mask - 1).astype(np.uint8)
        name = "%s_%s" % (path.split(os.sep)[-3].lower(), os.path.splitext(os.path.basename(path))[0])
        for r, c in grid(size[1], size[0], args.size):
            yield ("%s_%d_%d" % (name, r, c), image[r:r + args.size, c:c + args.size],
                   mask[r:r + args.size, c:c + args.size])


def main():
    args = parse_args()
    for folder in ["images"] + (["labels"] if args.labels else []):
        os.makedirs(os.path.join(args.out, folder), exist_ok=True)
    crops = chesapeake_crops if args.dataset == "chesapeake" else loveda_crops
    kept = 0
    for name, image, label in crops(args, random.Random(args.seed)):
        missing = (image == 0).all(-1)
        if args.labels:
            missing |= label == IGNORE
        if missing.mean() > args.max_nodata:
            continue
        Image.fromarray(image).save(os.path.join(args.out, "images", name + ".jpg"), quality=95)
        if args.labels:
            Image.fromarray(label).save(os.path.join(args.out, "labels", name + ".png"))
        kept += 1
        if kept == args.max_crops:
            break
    classes = CHESAPEAKE_CLASSES if args.dataset == "chesapeake" else LOVEDA_CLASSES
    save_json({"dataset": args.dataset, "crops": kept, "classes": classes, "args": vars(args)},
              os.path.join(args.out, "meta.json"))
    print("Saved %d crops to %s" % (kept, args.out))


if __name__ == "__main__":
    main()