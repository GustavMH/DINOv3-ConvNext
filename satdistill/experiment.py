import argparse
from pathlib import Path
import lightly_train
import torch
from satdistill import distill

RESULTS_DIR = Path("results/full100")
CROPS_DIR = Path("data/crops")
POINTS = (5, 10, 20, 40, 60, 80, 100)


def normalization(teacher):
    return distill.SAT_NORMALIZE if teacher == "sat" else distill.IMAGENET_NORMALIZE


def train(teacher, accelerator, devices, precision):
    """Distill one student for 100 epochs, saving a checkpoint every five epochs."""
    checkpoint_settings = {
        "every_n_epochs": 5,
        "save_top_k": -1,
        "filename": "epoch{epoch:03d}-step{step}",
    }
    lightly_train.pretrain(
        out=str(RESULTS_DIR / teacher / "training"),
        data=str(CROPS_DIR / "distill/images"),
        model=distill.STUDENT,
        method="distillation",
        method_args={"teacher": distill.TEACHERS[teacher]},
        transform_args={"normalize": normalization(teacher)},
        epochs=100,
        batch_size=64,
        num_workers=4,
        seed=0,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks={"model_checkpoint": checkpoint_settings},
    )


def export_student(teacher, epoch):
    """Export a saved training checkpoint for the baseline probe."""
    run = RESULTS_DIR / teacher
    pattern = f"epochepoch={epoch - 1:03d}-stepstep=*.ckpt"
    (checkpoint,) = (run / "training/checkpoints").glob(pattern)

    exported_weights = run / f"exported_epoch{epoch}.pt"
    lightly_train.export(out=str(exported_weights), checkpoint=str(checkpoint), overwrite=True)

    student_path = run / f"student_{teacher}_epoch{epoch:03d}.pt"
    torch.save(
        {
            "student": distill.export_to_timm(exported_weights),
            "student_name": distill.STUDENT_TIMM,
            "normalize": normalization(teacher),
        },
        student_path,
    )
    return student_path 


def export_checkpoints(teacher):
    for epoch in POINTS:
        export_student(teacher, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "export"))
    parser.add_argument("--teacher", choices=("web", "sat"), required=True)
    parser.add_argument("--crops", type=Path, default=CROPS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="32-true")
    args = parser.parse_args()
    CROPS_DIR = args.crops
    RESULTS_DIR = args.out

    if args.action == "train":
        train(args.teacher, args.accelerator, args.devices, args.precision)
    export_checkpoints(args.teacher)
