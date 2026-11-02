"""Distil a frozen DINOv3 ViT-L into the DINOv3 ConvNeXt-T with LightlyTrain and save the student in timm format.
python -m satdistill.distill --teacher sat --method v3 --data data/crops/distill
"""

import argparse
import logging
import os

import lightly_train
import timm
import torch
from torch import nn
from timm.models.convnext import checkpoint_filter_fn
import torch.nn.functional as F

from types import MethodType

TEACHERS = {"sat": "dinov3/vitl16-sat493m", "web": "dinov3/vitl16"}
METHODS = {"v3": "distillation", "v2": "distillationv2"}
SAT_NORMALIZE = {"mean": [0.430, 0.411, 0.296], "std": [0.213, 0.156, 0.143]}
IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def forward_features(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through feature extraction layers."""
    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    x = self.stem(x)
    x = self.stages(x)
    x = self.norm_pre(x)
    return x

class SatNormalizationFilter(logging.Filter):
    """LightlyTrain assumes ImageNet statistics for every DINOv3 teacher, the SAT-493M models use their own."""

    def filter(self, record):
        return (
            "expects LightlyTrain's ImageNet normalization" not in record.getMessage()
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--teacher", required=True, help="sat, web or a LightlyTrain model name"
    )
    parser.add_argument(
        "--student-size",
        required=True,
        help="tiny|small|base|large|xlarge",
    )
    parser.add_argument("--method", choices=list(METHODS), default="v3")
    parser.add_argument("--data", default="data/crops/distill")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def export_to_timm(exported_path, student_timm):
    model = timm.create_model(student_timm, pretrained=False, num_classes=0)
    state = checkpoint_filter_fn(
        torch.load(exported_path, map_location="cpu", weights_only=True), model
    )
    model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict()})
    return model.state_dict()


def main():
    args = parse_args()
    teacher = TEACHERS.get(args.teacher, args.teacher)
    normalize = SAT_NORMALIZE if "sat493m" in teacher else IMAGENET_NORMALIZE
    student_size = args.student_size
    student = f"dinov3/convnext-{student_size}"
    student_timm = f"convnext_{student_size}.dinov3_lvd1689m"
    name = f"convnext_{student_size}-web_from_{args.teacher.replace('/', '-')}_{args.method}"
    if args.tag:
        name += "_" + args.tag
    out_path = os.path.join(args.out_dir, f"student_{name}.pt")
    if os.path.exists(out_path):
        print("Already done:", out_path)
        return

    logging.getLogger("lightly_train._commands.train_helpers").addFilter(
        SatNormalizationFilter()
    )
    run_dir = os.path.join(args.out_dir, "lightly_" + name)
    resume = os.path.exists(os.path.join(run_dir, "checkpoints", "last.ckpt"))
    model = timm.create_model(f"convnext_{student_size}", pretrained=False, num_classes=0)
    model.forward_features = MethodType(forward_features, model)
    lightly_train.pretrain(
        out=run_dir,
        data=os.path.join(args.data, "images"),
        model=model,
        method=METHODS[args.method],
        method_args={"teacher": teacher},
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.workers,
        transform_args={"normalize": normalize},
        resume_interrupted=resume,
        overwrite=not resume,
        callbacks={"model_checkpoint": {"every_n_epochs": 10}}
    )

    student = export_to_timm(
        os.path.join(run_dir, "exported_models", "exported_last.pt"),
        student_timm
    )
    torch.save(
        {
            "student": student,
            "student_name": student_timm,
            "teacher": teacher,
            "method": args.method,
            "normalize": normalize,
            "args": vars(args),
        },
        out_path,
    )
    print("Saved", out_path)


if __name__ == "__main__":
    main()
