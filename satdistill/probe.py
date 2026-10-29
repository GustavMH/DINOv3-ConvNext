"""Linear segmentation probe on frozen last-layer features (batch norm + 1x1 convolution, as in the DINOv2/DINOv3
linear evaluation). Reports mIoU and pixel accuracy on each test folder.

python -m satdistill.probe --name chesapeake --model convnext_tiny.dinov3_lvd1689m \
    --train data/crops/ches_md_train --test data/crops/ches_md_test data/crops/ches_ny_test
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from satdistill.data import Crops
from satdistill.models import Backbone
from satdistill.prepare import IGNORE
from satdistill.utils import save_json, set_seed, short_name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="name of the probe task, e.g. chesapeake or loveda")
    parser.add_argument("--model", help="timm model name (released weights)")
    parser.add_argument("--ckpt", help="student file written by satdistill.distill (instead of --model)")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    if (args.model is None) == (args.ckpt is None):
        parser.error("give exactly one of --model and --ckpt")
    return args


@torch.no_grad()
def extract(backbone, folder, device, batch_size=64):
    loader = torch.utils.data.DataLoader(Crops(folder), batch_size=batch_size, num_workers=4)
    features, labels = [], []
    for x, y in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features.append(backbone.dense_features(x.to(device)).half().cpu())
        labels.append(y.to(torch.uint8))
    return torch.cat(features), torch.cat(labels)


def logits_of(head, features, size):
    return F.interpolate(head(features.float()), size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(head, features, labels, num_classes, device, batch_size=64):
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for start in range(0, len(features), batch_size):
        y = labels[start:start + batch_size].long().to(device)
        pred = logits_of(head, features[start:start + batch_size].to(device), y.shape[-2:]).argmax(1)
        valid = y != IGNORE
        confusion += torch.bincount(y[valid] * num_classes + pred[valid],
                                    minlength=num_classes ** 2).view(num_classes, num_classes).cpu()
    confusion = confusion.double()
    tp = confusion.diag()
    present = confusion.sum(1) > 0
    iou = tp / (confusion.sum(0) + confusion.sum(1) - tp).clamp(min=1)
    return {"mIoU": iou[present].mean().item(), "pixel_accuracy": (tp.sum() / confusion.sum()).item(),
            "IoU": [round(v, 4) if p else None for v, p in zip(iou.tolist(), present.tolist())]}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    if args.ckpt:
        backbone = Backbone(checkpoint=args.ckpt)
        model_name = os.path.basename(args.ckpt)[len("student_"):-len(".pt")]
    else:
        backbone = Backbone(args.model)
        model_name = short_name(args.model)
    backbone = backbone.eval().requires_grad_(False).to(device)
    with open(os.path.join(args.train, "meta.json")) as f:
        classes = json.load(f)["classes"]
    print("Probe %s | %s | %s" % (args.name, model_name, device))

    start = time.time()
    train_x, train_y = extract(backbone, args.train, device)
    tests = {os.path.basename(os.path.normpath(t)): extract(backbone, t, device) for t in args.test}
    print("features %s computed in %.0f s" % (tuple(train_x.shape), time.time() - start))

    head = nn.Sequential(nn.BatchNorm2d(train_x.shape[1]), nn.Conv2d(train_x.shape[1], len(classes), 1)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(train_x) / args.batch_size))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for epoch in range(args.epochs):
        head.train()
        order, total = torch.randperm(len(train_x)), 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            y = train_y[idx].long().to(device)
            loss = F.cross_entropy(logits_of(head, train_x[idx].to(device), y.shape[-2:]), y, ignore_index=IGNORE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print("epoch %d | train loss %.4f" % (epoch + 1, total / len(order)))

    head.eval()
    results = {name: evaluate(head, x, y, len(classes), device) for name, (x, y) in tests.items()}
    results["train"] = evaluate(head, train_x, train_y, len(classes), device)
    for name, r in results.items():
        print("%-20s mIoU %.4f | pixel accuracy %.4f | IoU %s" % (name, r["mIoU"], r["pixel_accuracy"], r["IoU"]))
    save_json({"probe": args.name, "model": model_name, "classes": classes, "feature_shape": list(train_x.shape[1:]),
               "n_train": len(train_x), "results": results, "args": vars(args)},
              os.path.join(args.out_dir, "probe_%s_%s.json" % (args.name, model_name)))


if __name__ == "__main__":
    main()
