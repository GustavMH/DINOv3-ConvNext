"""Distil a frozen DINOv3 ViT-L into the DINOv3 ConvNeXt-T with LightlyTrain and save the student in timm format.
python -m satdistill.distill --teacher sat --method v3 --data data/crops/distill
"""
import argparse
import json
import logging
import os

import lightly_train
import timm
import torch
from timm.models.convnext import checkpoint_filter_fn

TEACHERS = {"sat": "dinov3/vitl16-sat493m", "web": "dinov3/vitl16"}
METHODS = {"v3": "distillation", "v2": "distillationv2"}
STUDENT = "dinov3/convnext-tiny"
STUDENT_TIMM = "convnext_tiny.dinov3_lvd1689m"
# timm name used to re-instantiate whatever the student was, so the export and the probe agree
STUDENT_TIMM_BY_ARCH = {
    "convnext_tiny": "convnext_tiny.dinov3_lvd1689m",
    "convnext_small": "convnext_small.dinov3_lvd1689m",
    "convnext_base": "convnext_base.dinov3_lvd1689m",
    "convnext_large": "convnext_large.dinov3_lvd1689m",
}
SAT_NORMALIZE = {"mean": [0.430, 0.411, 0.296], "std": [0.213, 0.156, 0.143]}
IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class SatNormalizationFilter(logging.Filter):
    """LightlyTrain assumes ImageNet statistics for every DINOv3 teacher, the SAT-493M models use their own."""

    def filter(self, record):
        return "expects LightlyTrain's ImageNet normalization" not in record.getMessage()


def parse_method_arg(text):
    """Parse a repeatable --method-arg KEY=VALUE (JSON values allowed: 0, 1.0, true, "x")."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError("expected KEY=VALUE, got %r" % text)
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass  # keep the raw string
    return key, value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, help="sat, web or a LightlyTrain model name")
    parser.add_argument("--method", choices=list(METHODS), default="v3")
    parser.add_argument("--student", default=STUDENT, help="LightlyTrain student model (default %(default)s)")
    parser.add_argument("--data", default="data/crops/distill")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=0, help="training seed (use >0 for noise-floor repeats)")
    parser.add_argument("--method-arg", metavar="KEY=VALUE", action="append", type=parse_method_arg,
                        help="override a method argument, e.g. --method-arg loss_local_weight=0 "
                             "--method-arg queue_size=0 (repeatable)")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def student_timm_name(student):
    """'dinov3/convnext-small' or 'timm/convnext_small' -> 'convnext_small.dinov3_lvd1689m'."""
    arch = student.split("/")[-1].replace("-", "_")
    return STUDENT_TIMM_BY_ARCH.get(arch, STUDENT_TIMM)


def export_to_timm(exported_path, timm_name):
    model = timm.create_model(timm_name, pretrained=False, num_classes=0)
    state = checkpoint_filter_fn(torch.load(exported_path, map_location="cpu", weights_only=True), model)
    # Fail loudly: silently dropping keys would train a good student and export half of it,
    # and every downstream probe would look plausibly bad.
    missing = sorted(set(model.state_dict()) - set(state))
    assert not missing, "export incomplete: %d key(s) missing, e.g. %s" % (len(missing), missing[:3])
    model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict()})
    return model.state_dict()


def main():
    args = parse_args()
    teacher = TEACHERS.get(args.teacher, args.teacher)
    normalize = SAT_NORMALIZE if "sat493m" in teacher else IMAGENET_NORMALIZE
    student_tag = "convnext_tiny-web" if args.student == STUDENT else args.student.replace("/", "-")
    timm_name = student_timm_name(args.student)
    name = "%s_from_%s_%s" % (student_tag, args.teacher.replace("/", "-"), args.method)
    if args.tag:
        name += "_" + args.tag
    method_args = {"teacher": teacher}
    method_args.update(dict(args.method_arg or []))
    out_path = os.path.join(args.out_dir, "student_%s.pt" % name)
    if os.path.exists(out_path):
        print("Already done:", out_path)
        return

    logging.getLogger("lightly_train._commands.train_helpers").addFilter(SatNormalizationFilter())
    run_dir = os.path.join(args.out_dir, "lightly_" + name)
    resume = os.path.exists(os.path.join(run_dir, "checkpoints", "last.ckpt"))
    lightly_train.pretrain(
        out=run_dir, data=os.path.join(args.data, "images"), model=args.student, method=METHODS[args.method],
        method_args=method_args, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.workers,
        transform_args={"normalize": normalize}, resume_interrupted=resume, overwrite=not resume,
        seed=args.seed)

    student = export_to_timm(os.path.join(run_dir, "exported_models", "exported_last.pt"), timm_name)
    torch.save({"student": student, "student_name": timm_name, "teacher": teacher, "method": args.method,
                "normalize": normalize, "args": vars(args)}, out_path)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
