#Evaluate seven student checkpoints per teacher with three probe seeds.
import subprocess
from pathlib import Path


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


def evaluate():
    probe = ["python", "-u", "-m", "satdistill.probe"]

    for teacher, epoch in MODELS:
        if epoch == 0:
            model = ["--model", "convnext_tiny.dinov3_lvd1689m"]
        else:
            student = ROOT / teacher / f"student_{teacher}_epoch{epoch:03d}.pt"
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
