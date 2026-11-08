#Evaluate seven student checkpoints per teacher with three probe seeds.
import subprocess
from pathlib import Path
import lightly_train
import torch
from satdistill import distill


ROOT = Path("results/full100")
OUTPUT = ROOT / "training_length_ablation"
CROPS = Path("data/crops")
POINTS = (5, 10, 20, 40, 60, 80, 100)
TASKS = [
    ("chesapeake", "ches_md_train", ["ches_md_test", "ches_ny_test"]),
    ("loveda", "loveda_train", ["loveda_val"]),
]


SEEDS = (0, 1, 2)
MODELS = [("released", 0)] + [
    (teacher, epoch) for teacher in ("sat", "web") for epoch in POINTS
]


def student_checkpoint(teacher, epoch):
    student = ROOT / teacher / f"student_{teacher}_epoch{epoch:03d}.pt"
    if student.exists():
        return student

    run = ROOT / f"lightly_convnext_tiny-web_from_{teacher}_v3"
    checkpoint, = (run / "checkpoints").glob(
        f"epochepoch={epoch - 1:03d}-stepstep=*.ckpt"
    )
    student.parent.mkdir(parents=True, exist_ok=True)
    exported = student.parent / f"exported_epoch{epoch}.pt"
    lightly_train.export(out=str(exported), checkpoint=str(checkpoint), overwrite=True)
    torch.save({
        "student": distill.export_to_timm(exported),
        "student_name": distill.STUDENT_TIMM,
        "normalize": distill.SAT_NORMALIZE if teacher == "sat" else distill.IMAGENET_NORMALIZE,
    }, student)
    return student


def evaluate():
    probe = ["python", "-u", "-m", "satdistill.probe"]

    for teacher, epoch in MODELS:
        if epoch == 0:
            model = ["--model", "convnext_tiny.dinov3_lvd1689m"]
        else:
            student = student_checkpoint(teacher, epoch)
            model = ["--ckpt", str(student)]

        for seed in SEEDS:
            for task, train, tests in TASKS:
                subprocess.run([
                    *probe, *model,
                    "--name", task,
                    "--train", str(CROPS / train),
                    "--test", *(str(CROPS / split) for split in tests),
                    "--seed", str(seed),
                    "--out-dir", str(OUTPUT / f"{teacher}_{epoch:03d}" / f"seed{seed}" / task),
                ], check=True)


if __name__ == "__main__":
    evaluate()
