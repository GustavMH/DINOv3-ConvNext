# Training-length ablation

Evaluates ConvNeXt-Tiny checkpoints at epochs 5, 10, 20, 40, 60, 80 and 100 with three probe seeds on Maryland, New York and LoveDA.

Run from this folder. Requires prepared crops in data/crops and exported student weights in results/full100/web and results/full100/sat, named student_web_epoch005.pt, student_sat_epoch005.pt, and so on.

## Install

```bash
python -m pip install -r requirements.txt
```

## Evaluate

```bash
python -m satdistill.probe_all_checkpoints
```

Scores are saved as JSON files in results/full100/training_length_ablation, grouped by model, epoch, seed and dataset.
