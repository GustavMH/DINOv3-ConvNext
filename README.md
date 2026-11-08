# Training-length ablation

Train ConvNeXt-Tiny with web and satellite teachers for 100 epochs, then evaluate seven checkpoints with three probe seeds.

## Install and prepare

```bash
python -m pip install -r requirements.txt
bash scripts/download.sh chesapeake
bash scripts/download.sh loveda
bash scripts/run_pilot.sh prepare
```

## Train

```bash
python -m satdistill.distill --teacher web --epochs 100 --workers 4 --out-dir results/full100 --save-checkpoints
python -m satdistill.distill --teacher sat --epochs 100 --workers 4 --out-dir results/full100 --save-checkpoints
```

## Evaluate

```bash
python -m satdistill.probe_all_checkpoints
```

The script exports epochs 5, 10, 20, 40, 60, 80 and 100, then evaluates each with probe seeds 0, 1 and 2.

Probes fit on Maryland train and score on Maryland test and New York test. A separate probe fits on LoveDA train and scores on LoveDA validation.

JSON scores are saved under results/full100/training_length_ablation.

Scores can vary with software versions, hardware, precision and random initialisation.
